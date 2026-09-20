package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

func TestGatewayRoutesOnlyWorkerSurface(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-worker" {
			t.Error("credential not preserved")
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	handler, err := gatewayHandler(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/", "/auth/config", "/v1/tasks", "/v1/workers", "/%76%31/worker", "/healthz", "/v1/worker"} {
		request := httptest.NewRequest("GET", path, nil)
		request.Header.Set("Authorization", "Bearer test-worker")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		want := http.StatusNotFound
		if path == "/healthz" || path == "/v1/worker" {
			want = http.StatusNoContent
		}
		if response.Code != want {
			t.Errorf("%s: got %d, want %d", path, response.Code, want)
		}
	}
	for _, target := range []string{"http://example.com:8081", "http://127.0.0.1:8081/path", "https://127.0.0.1:8081", "http://user@127.0.0.1:8081", "http://127.0.0.1:" + strconv.Itoa(65536)} {
		if _, err := gatewayHandler(target); err == nil {
			t.Errorf("accepted %s", target)
		}
	}
}

func TestTargetRestrictedToTailnet(t *testing.T) {
	for _, target := range []string{"example.com:443", "127.0.0.1:80", "gateway.ts.net:0", "gateway.ts.net:65536", "gateway..ts.net:443", "x/evil.ts.net:443"} {
		if validateTarget(target) == nil {
			t.Errorf("accepted %q", target)
		}
	}
	if err := validateTarget("gateway.example.ts.net:8443"); err != nil {
		t.Fatal(err)
	}
}

func TestForwardBytesAndCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	targets := make(chan string, 1)
	upstream, peer := net.Pipe()
	defer peer.Close()
	finished := make(chan error, 1)
	go func() {
		finished <- serve(ctx, listener, "gateway.example.ts.net:8443", func(_ context.Context, network, target string) (net.Conn, error) {
			targets <- network + ":" + target
			return upstream, nil
		})
	}()
	client, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	client.SetDeadline(time.Now().Add(3 * time.Second))
	peer.SetDeadline(time.Now().Add(3 * time.Second))
	go func() { io.Copy(peer, peer) }()
	want := "opaque TLS bytes\x00\xff"
	if _, err := io.WriteString(client, want); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(want))
	if _, err := io.ReadFull(client, got); err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("changed stream: %q", got)
	}
	if got := <-targets; got != "tcp:gateway.example.ts.net:8443" {
		t.Fatal(got)
	}
	cancel()
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("active tunnel did not shut down")
	}
}

func TestServiceChannelsAreNarrowlyRouted(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Worker-Session") != "session-one" {
			t.Error("service identity not preserved")
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	handler, err := gatewayHandler(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/service-control/svc-one", "/v1/service-data/svc-one", "/v1/service-bundle/svc-one", "/internal/serve/svc-one/health", "/serve/svc-one/health", "/v1/service-data/", "/v1/service-data/../tasks", "/v1/service-data/%73vc-one", "/v1/service-data/svc-one/extra"} {
		for _, method := range []string{"GET", "POST"} {
			req := httptest.NewRequest(method, path, nil)
			req.Header.Set("X-Worker-Session", "session-one")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, req)
			want := http.StatusNotFound
			if method == "GET" && (path == "/v1/service-control/svc-one" || path == "/v1/service-data/svc-one" || path == "/v1/service-bundle/svc-one") {
				want = http.StatusNoContent
			}
			if response.Code != want {
				t.Errorf("%s %s: got %d, want %d", method, path, response.Code, want)
			}
		}
	}
}
