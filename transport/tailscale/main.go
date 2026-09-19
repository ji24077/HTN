// orchestrator-tunnel carries one worker's TLS connection through embedded Tailscale.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"tailscale.com/tsnet"
)

func validateTarget(target string) error {
	host, port, err := net.SplitHostPort(target)
	if err != nil {
		return errors.New("target must be a full .ts.net hostname and port")
	}
	p, err := strconv.Atoi(port)
	if err != nil || p < 1 || p > 65535 || !strings.HasSuffix(host, ".ts.net") || len(host) > 253 {
		return errors.New("target must be a full .ts.net hostname and valid port")
	}
	for _, label := range strings.Split(host, ".") {
		if label == "" || len(label) > 63 || strings.HasPrefix(label, "-") || strings.HasSuffix(label, "-") {
			return errors.New("invalid target hostname")
		}
		for _, c := range label {
			if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
				return errors.New("invalid target hostname")
			}
		}
	}
	return nil
}

type dialFunc func(context.Context, string, string) (net.Conn, error)

// Only a fixed destination is reachable. This is not a general HTTP/SOCKS proxy.
func forward(ctx context.Context, local net.Conn, target string, dial dialFunc) {
	defer local.Close()
	dialCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	remote, err := dial(dialCtx, "tcp", target)
	cancel()
	if err != nil {
		log.Print("private gateway connection failed")
		return
	}
	defer remote.Close()
	stop := context.AfterFunc(ctx, func() { local.Close(); remote.Close() })
	defer stop()
	done := make(chan struct{})
	go func() {
		defer close(done)
		io.Copy(remote, local)
		remote.Close()
	}()
	io.Copy(local, remote)
	local.Close()
	<-done
}

func serve(ctx context.Context, listener net.Listener, target string, dial dialFunc) error {
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(ctx, func() { listener.Close() })
	defer stop()
	var active sync.WaitGroup
	defer active.Wait()
	defer cancel()
	// Bound connections from other local processes as well as the worker.
	slots := make(chan struct{}, 8)
	for {
		conn, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		select {
		case slots <- struct{}{}:
			active.Add(1)
			go func() {
				defer active.Done()
				defer func() { <-slots }()
				forward(ctx, conn, target, dial)
			}()
		default:
			conn.Close()
		}
	}
}

func run() error {
	mode := flag.String("mode", "worker", "worker or gateway")
	upstream := flag.String("upstream", "http://127.0.0.1:8081", "gateway-only loopback HTTP listener")
	target := flag.String("target", "", "private gateway's full .ts.net hostname:port")
	hostname := flag.String("hostname", "", "unique worker name in Tailscale")
	state := flag.String("state-dir", "", "private, persistent node identity directory")
	flag.Parse()
	if *mode != "worker" && *mode != "gateway" {
		return errors.New("mode must be worker or gateway")
	}
	if *mode == "worker" {
		if err := validateTarget(*target); err != nil {
			return err
		}
	} else if _, err := gatewayHandler(*upstream); err != nil {
		return err
	}
	if *hostname == "" || *state == "" {
		return errors.New("hostname and state-dir are required")
	}
	if err := os.MkdirAll(*state, 0700); err != nil {
		return errors.New("cannot create node state directory")
	}
	if err := os.Chmod(*state, 0700); err != nil {
		return errors.New("cannot protect node state directory")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	// The Python parent keeps stdin open. Exit even if it is killed without cleanup.
	go func() { io.Copy(io.Discard, os.Stdin); cancel() }()
	srv := &tsnet.Server{
		Dir: *state, Hostname: *hostname, AuthKey: os.Getenv("TS_AUTHKEY"),
		UserLogf: log.Printf,
		Logf:     func(string, ...any) {},
	}
	defer srv.Close()
	upCtx, upCancel := context.WithCancel(ctx)
	if *mode == "worker" {
		upCancel()
		upCtx, upCancel = context.WithTimeout(ctx, 5*time.Minute)
	}
	status, err := srv.Up(upCtx)
	upCancel()
	if err != nil {
		return errors.New("Tailscale enrollment failed or timed out; check sign-in and device approval")
	}
	if *mode == "gateway" {
		listener, err := srv.ListenTLS("tcp", ":8443")
		if err != nil {
			return errors.New("enable MagicDNS and HTTPS certificates at https://login.tailscale.com/admin/dns, then restart the backend")
		}
		defer listener.Close()
		handler, _ := gatewayHandler(*upstream)
		httpServer := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
		stop := context.AfterFunc(ctx, func() { httpServer.Close() })
		defer stop()
		json.NewEncoder(os.Stdout).Encode(map[string]string{
			"event": "gateway_ready",
			"url":   "wss://" + strings.TrimSuffix(status.Self.DNSName, ".") + ":8443/v1/worker",
		})
		err = httpServer.Serve(listener)
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return errors.New("cannot open local tunnel socket")
	}
	defer listener.Close()
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
		"event": "ready", "port": listener.Addr().(*net.TCPAddr).Port,
	}); err != nil {
		return err
	}
	return serve(ctx, listener, *target, srv.Dial)
}

func gatewayHandler(upstream string) (http.Handler, error) {
	u, err := url.Parse(upstream)
	if err != nil || u.Scheme != "http" || u.Hostname() != "127.0.0.1" || u.User != nil ||
		u.Path != "" || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("gateway upstream must be http://127.0.0.1:PORT")
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil || p < 1 || p > 65535 {
		return nil, errors.New("invalid gateway upstream port")
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.Transport = &http.Transport{Proxy: nil, DialContext: (&net.Dialer{Timeout: 10 * time.Second}).DialContext}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		http.Error(w, "worker gateway unavailable", http.StatusBadGateway)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || (r.URL.EscapedPath() != "/v1/worker" && r.URL.EscapedPath() != "/healthz") {
			http.NotFound(w, r)
			return
		}
		proxy.ServeHTTP(w, r)
	}), nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "orchestrator-tunnel:", err)
		os.Exit(1)
	}
}
