# [팀명] — 팀 핸드북 (한국어 전체본)

**아무한테서나 GPU를 빌린다. 에이전트가 묶고 빠르게 만든다.**

> 영어판(`team-handbook.md`)과 같은 내용. 팀원 배포용은 영어판 또는 역할별 파일을 쓸 것.

| 파트 | 읽는 사람 | 내용 |
|---|---|---|
| **0. 시작** | 세 명 다 | 뭘 만드나, 계약, 블로커 세 개 |
| **1. Jack** | Jack | 네트워크 + 공유 마켓플레이스 |
| **2. Ethan** | Ethan | 훈련 실행 엔진 |
| **3. Ji** | Ji | 에이전트 + 최적화 + 대시보드 |
| **4. 통합** | 세 명 다 | 합치는 시점, 자를 순서 |
| **5. 데모** | 세 명 다 (특히 발표자) | 시연 순서, 폴백, Q&A |
| **A. 최적화 레버** | Ji | 규칙 표와 측정 설계 |

**아직 안 정해진 두 가지 — 오늘 정할 것:** 팀 이름, 그리고 렌탈이 크레딧인지 실제 돈인지. 이 문서는 **크레딧**으로 가정한다.

---
---

# 파트 0 — 시작

## 뭘 만드나

**사람들이 서로의 놀고 있는 GPU를 빌려주고 빌려 쓰는 시장 + 빌린 칩을 묶어서 잘 돌려주는 에이전트.**

두 겹이고, 붙어 있다는 게 전부다.

```
Vast.ai            칩을 판다.        묶는 건 네 문제.
Ray / torchft      묶어준다.         찾는 건 네 문제.
우리                둘 다 — 게다가 뭘 빌릴지까지 에이전트가 고른다.
```

민지는 GPU가 없다. 친구 셋에게서 빌린다. 그다음엔 할 일이 없다 — 에이전트가 묶고, 설정을 고르고, 더 싼 칩이 비면 옮긴다. 준호는 반대편에서 수업 간 사이 노는 4090으로 크레딧을 번다.

## 세 가지 메커니즘

피치덱엔 기능이 열두 개지만, 밑바닥엔 **메커니즘이 셋뿐이다.**

1. **칩을 확보하는 것** — 내 풀에 누가 있나 (빌리기, 반납, 크레딧)
2. **워커 집합을 바꾸는 것** — 동기화 시점에 pod 안에 누가 있나 (합류, 이탈, 마이그레이션, 낙오자 제외)
3. **설정을 바꾸는 것** — 재보고 파라미터를 다시 고르기

1과 2는 사실상 같은 코드다 — 칩을 빌리는 것과 워커가 pod에 합류하는 것은 범위만 다른 같은 사건이다. 모든 최적화 레버는 3번이다.

**그리고 이게 에이전트 두 단과 1:1로 대응한다.**

```
라우터        워커 집합을 결정  → 메커니즘 1, 2
칩 에이전트    설정을 결정      → 메커니즘 3
```

이걸 기억하면 코드베이스가 작게 유지된다.

## 에이전트, 두 단

차별점이니 정확하게 말할 것.

```
라우터  (풀 전체를 봄)
  · 뭘 빌릴까 — 충분한 것 중 제일 싼 것, 구세대 포함
  · 언제 어디로 옮길까
  · 느려서 뺄 사람은 누구
  · H(동기화 주기) — 측정된 네트워크 상태에 달림
        ↓ 위임
칩 에이전트  (칩 종류마다 하나)
  Ampere24GB(3090) · Ada24GB(4090) · MI250 · V100 · Default
  · 이 칩에서의 dtype, 배치, attention, checkpointing
  · "나는 3090을 안다"
```

**칩별로 나누는 게 왜 방어 가능한가:** 칩 지식은 경험적이고 이전이 안 된다. 3090에서 통하는 것(24GB, Ampere, NVLink 없음)과 MI250에서 통하는 건 다르고, 스펙만으로 유도할 수 없어서 돌려봐야 안다. **그리고 이게 데이터 해자와 정확히 맞물린다:** 실행 기록이 칩 종류마다 따로 쌓이고, 3090 에이전트는 3090 데이터로만 똑똑해진다. 새 칩이 오면 기본값에서 시작하는 새 에이전트가 생기고 학습한다.

**P0에서 이것들은 규칙 기반 Python 클래스다.** 괜찮다 — 다만 **구조는 진짜로 만들 것**(파트 3, S-4). "칩마다 전용 에이전트가 있다"가 코드상 사실이 아니면 심사위원이 코드를 열었을 때 역효과가 난다.

## 핵심 설계 규칙

| 상황 | 방식 |
|---|---|
| 같은 벤더 (NVIDIA↔NVIDIA, AMD↔AMD) | **동시 사용** — 분산 훈련 + 추론 |
| 다른 벤더 (NVIDIA↔AMD) | **마이그레이션만** — 작업을 통째로 이동 |

CUDA와 ROCm을 한 훈련 안에서 엮으려 하지 않는다. 이 선을 그은 게 프로젝트를 만들 수 있게 만든다.

**그리고 이렇게 말할 것:** 우리는 "코드를 변환"하지 않는다. 같은 PyTorch 코드가 양 벤더에서 이미 돈다(CUDA 빌드, ROCm 빌드). **움직이는 건 가중치 파일 하나뿐이다.** "에이전트가 코드를 마이그레이션한다"고 하면 심사위원은 "CUDA 커널을 ROCm으로 트랜스파일한다"로 알아듣고, 그건 사실이 아니라서 거기서 무너진다.

## 크레딧, 현금 아님 (가정)

렌탈은 **크레딧**으로 정산한다. GPU-시간을 기여하고 나중에 쓴다. 지급도, 분쟁도, KYC도 없다. "놀 때 빌려주고 필요할 때 빌려 쓴다."

실제 돈으로 가기로 했다면 T+3 전에 전원에게 알릴 것 — 정산·신뢰 점수·분쟁 처리가 붙고 그건 해커톤 범위 밖이다. **심사위원이 반드시 묻는다. 답을 준비해둘 것.**

## 분담

| | 담당 | 티켓 | 부하 |
|---|---|---|---|
| **Jack** | 네트워크 + 공유 마켓플레이스 | N-1…N-6, S-1, S-8 | ~12h |
| **Ethan** | 훈련 실행 엔진 | T-1…T-8 | ~13h |
| **Ji** | 에이전트 + 최적화 + 대시보드 | S-2…S-7, S-9 | ~10.5h |

**경계 한 줄:** Ji가 결정하고, Ethan이 실행하고, Jack이 그 사이를 연결한다.

서로 안 막히는 규칙 셋:

- Ji는 훈련 루프 코드를 건드리지 않는다.
- Ethan은 규칙을 하드코딩하지 않는다. 설정은 JSON으로 받아서 실행한다.
- **"스케줄러"라는 컴포넌트는 없다.** 스케줄링, 칩 선택, 렌탈 결정, 최적화는 전부 하나이고 이름은 **에이전트**(라우터 + 칩 에이전트)다. 두 이름을 쓰면 두 번 만들거나 아무도 안 만든다.

## 기술 스택

| 레이어 | 선택 | 폴백 |
|---|---|---|
| 분산 훈련 | torchft (DiLoCo + fault tolerance) | diloco_simple / 서버 중계 평균 |
| 추론 | vLLM | — |
| 네트워크 | Tailscale (WireGuard 오버레이) | — |
| 집합 통신 백엔드 | **gloo** (NCCL 절대 안 됨) | — |
| 마이그레이션 | safetensors | — |
| 프레임워크 | PyTorch (CUDA + ROCm) | — |
| 서버 | FastAPI + WebSocket | — |
| 대시보드 | 단일 HTML + WebSocket, 또는 React + Recharts | — |
| 에이전트 | 규칙 기반 Python 클래스 | P2에서 칩별 학습 모델 |
| DB | SQLite | Postgres |

---

## HOUR 0 — 계약 고정 (세 명 다, 60분)

**건너뛰지 말 것.** 이거 없이 흩어지면 세 코드베이스가 안 붙고, 그걸 14시간째에 알게 된다. 여기 한 시간이 나중에 열두 시간을 번다.

한 화면 보면서 아래를 합의하고 `contracts.py`로 커밋한다. 이후 변경은 세 명 합의 필요.

### 계약 1 — 워커 → 서버 이벤트 (WebSocket JSON)

```json
{"type":"worker.register","worker_id":"w1","owner":"junho","vendor":"nvidia",
 "gpu":"RTX 3090","vram_gb":24,"cc":8.6,"chip_class":"ampere_24gb",
 "share":{"enabled":true,"not_gaming":true,"always_between":["00:00","08:00"]},
 "credits_per_hour":1.0,"ts":1737000000.0}

{"type":"worker.heartbeat","worker_id":"w1","gpu_util":0.93,"mem_used_gb":21.2,"ts":...}

{"type":"train.step","worker_id":"w1","step":1234,"loss":1.83,
 "step_time_s":0.21,"tokens":32768,"ts":...}

{"type":"train.sync","round":7,"participants":["w1","w2"],"bytes":41943040,
 "duration_s":2.1,"global_loss":1.79,"ts":...}

{"type":"worker.left","worker_id":"w1","reason":"timeout|user|game","ts":...}

{"type":"credits.update","owner":"junho","earned_gpu_hours":3.2,"balance":128.5,"ts":...}

{"type":"migration.progress","job_id":"j1","from":"w1","to":"w3",
 "phase":"waiting_sync|saving|transferring|loading|resumed","pct":0.4,"ts":...}

{"type":"probe.result","worker_id":"w1","chip_class":"ampere_24gb",
 "config_name":"baseline|optimized","t_step_median_s":2.31,
 "tokens_per_s":14200,"peak_vram_gb":9.1,"gpu_util":0.62,"t_sync_s":2.0}
```

### 계약 2 — 서버 → 워커 명령

```json
{"type":"job.start","job_id":"j1","config":{ ...계약 3... }}
{"type":"job.stop","job_id":"j1"}
{"type":"checkpoint.save","job_id":"j1","round":7}
{"type":"checkpoint.load","job_id":"j1","uri":"file:///.../round_7.safetensors"}
{"type":"config.update","micro_batch":32}
```

### 계약 3 — 잡 설정 (라우터의 출력 = Ethan의 입력)

```json
{"job_id":"j1","model":"nanogpt-124m","dtype":"bf16","attention":"sdpa",
 "micro_batch":32,"grad_accum":1,"global_batch_tokens":32768,
 "H":190,"backend":"gloo","workers":["w1","w2"],"total_steps":8000}
```

**이 JSON 하나가 Ji와 Ethan 사이의 인터페이스 전부다.**

### 계약 4 — 체크포인트 레이아웃

```
ckpt/{job_id}/round_{N}.safetensors   ← 가중치만
ckpt/{job_id}/round_{N}.meta.json     ← {round, global_step, global_loss, config_hash, ts}
```

### 계약 5 — `chip_class` 문자열

세 명이 이걸 키로 쓴다. 정확한 문자열을 지금 합의할 것.

```
ampere_24gb   (RTX 3090)
ada_24gb      (RTX 4090)
turing_16gb   (V100 등)
cdna_amd      (MI250, 7900)
default       (모르는 칩 — 콜드 스타트)
```

Jack이 `worker.register`와 SQLite에 쓰고, Ethan이 프로브에 태깅하고, Ji가 라우터를 여기에 디스패치한다. **셋이 다른 문자열을 쓰면 칩별 학습이 에러 없이 조용히 안 된다.**

**완료 기준:** 세 레포가 같은 `contracts.py`를 import하고, 각자 그걸로 더미 이벤트를 한 번씩 만들어봤다.

---

## 블로커 세 개

이게 안 풀리면 나머지가 무의미하다. **T+3까지 답이 나와야 한다.**

| ID | 담당 | 내용 | 실패하면 |
|---|---|---|---|
| **N-1** | Jack | Tailscale 피어가 `relay`가 아닌 `direct` | 릴레이면 2~35Mbit/s. H를 올리고 모델 축소 |
| **N-2** | Jack | gloo all-reduce가 머신 간에 통과 | 계획 변경 — 한 머신 2프로세스, 또는 서버 중계 평균 |
| **T-2a** | Ethan | torchft nightly가 여기서 실제로 도는지 | diloco_simple로 후퇴 → T-3이 2h → 5h |

> **Ethan은 T-2a를 지금 당장, 자기 목록의 다른 어떤 것보다 먼저 돌린다.**

**미리 알아둘 것: NCCL은 NAT를 사이에 둔 Tailscale 위에서 동작하지 않는다.** 문서화된 사실이다(NVIDIA/nccl #1606, "not planned"로 종료). 시간 쓰지 말 것. gloo를 쓰고, DiLoCo가 동기화를 드물게 하는 게 그걸 괜찮게 만든다. 타협이 아니라 **DiLoCo를 고른 이유** 자체다.

---
---

# 파트 1 — JACK: 네트워크 + 공유 마켓플레이스

> "흩어진 기기를 연결하고, 말이 통하게 하고, 주인이 빌려줄 수 있게 한다."

Jack은 **양쪽 끝을 다 갖는다.** 각 GPU 머신의 워커 데몬과, 모두가 접속하는 서버 허브. 같은 프로토콜의 양쪽이라 한 사람이 짜야 프로토콜 버그가 안 생긴다. 마켓플레이스의 공급 측도 Jack이다 — 공유 규칙, 즉시 회수, 크레딧.

## 유저 스토리

민지가 로그인하면 **친구 A의 3090, 친구 B의 3090, 랩의 AMD 7900이 각각 크레딧 단가와 함께 목록으로 뜬다.** 세 아파트, 세 공유기. 민지 화면에는 그냥 메뉴다.

준호는 앱을 깔고 "게임 안 할 때만"을 설정하고 잊는다. 수업 간 사이 4090이 일하며 크레딧을 벌고, 게임을 켜면 즉시 빠진다. **준호 쪽에선 게임이 안 끊기고, 민지 쪽에선 워커 하나가 회색이 된다.** 같은 사건인데 양쪽 다 불편하지 않다 — 이게 Jack이 만드는 것이다.

나중에 준호가 자기 프로젝트에 GPU 여덟 대가 필요할 때 쌓아둔 크레딧을 쓴다.

## 핵심 결정

| 항목 | 선택 | 이유 |
|---|---|---|
| 오버레이 네트워크 | Tailscale (WireGuard) | NAT 뒤 개인 기기를 묶는 가장 빠른 길 |
| 집합 통신 백엔드 | **gloo** | NCCL은 NAT 너머에서 실패. 시도하지 말 것 |
| 서버 | FastAPI + WebSocket | 나머지가 다 Python |
| 체크포인트 전송 | HTTP multipart + sha256 | 단순함이 최고. P2P는 과함 |
| 정산 | **크레딧**, 현금 아님 | 지급·분쟁·KYC 없음 |
| DB | SQLite | 지금은 충분 |

**Jack이 무대에서 답해야 할 질문:** *"NCCL이 NAT 넘어서 됩니까?"*
→ "안 됩니다. 그래서 gloo를 씁니다. NCCL은 스텝마다 통신하는 걸 전제로 최적화된 백엔드인데, 저희는 H스텝에 한 번만 동기화하니 CPU 측 collective로 충분합니다. **NCCL이 못 가는 네트워크를 쓸 수 있게 만드는 게 DiLoCo를 고른 이유입니다.**"

## 티켓

| ID | 티켓 | 시간 | P |
|---|---|---|---|
| N-1 | Tailscale 메시 + direct 확인 | 0.5h | **P0 ★블로커** |
| N-2 | gloo all-reduce 머신 간 통과 | 1h | **P0 ★블로커** |
| N-3 | 워커 데몬 | 2h | P0 |
| S-1 | FastAPI + WebSocket 허브 | 1.5h | P0 |
| N-4 | 체크포인트 전송 | 2h | P0 |
| N-5 | 워커 회수/재합류 CLI | 1h | P0 |
| **N-6a** | **공유 규칙 + 크레딧 원장** | **0.75h** | **P0** ← P1이었음 |
| S-8 | SQLite 실행 로깅 | 1h | P0 |
| N-6b | 자동 게임 감지 | 1.25h | P1 |

### N-1 · Tailscale 메시 + direct 확인 (0.5h) ★블로커

```bash
# scripts/netcheck.sh
tailscale status --json | jq -r '.Peer[] | "\(.HostName) \(.CurAddr) \(if .Relay=="" then "DIRECT" else "RELAY:"+.Relay end)"'
iperf3 -c <peer-100.x-ip> -t 5     # 실측 대역폭
```

- **`RELAY`는 비상이다.** DERP 릴레이는 2~35Mbit/s로 주저앉는다. Ji에게 즉시 알려 라우터가 H를 올리게 한다.
- **현장에서 다시 돌릴 것.** 행사장 와이파이는 클라이언트 격리를 거는 경우가 많아, 집에서 direct였어도 현장에서 relay로 떨어질 수 있다.

**완료 기준:** 피어별 direct/relay와 실측 대역폭 표, 명령 한 방으로 재현 가능.

### N-2 · gloo all-reduce 머신 간 통과 (1h) ★블로커

```python
# scripts/gloo_check.py
import os, time, torch, torch.distributed as dist
os.environ["GLOO_SOCKET_IFNAME"] = "tailscale0"
dist.init_process_group("gloo",
    init_method=f"tcp://{MASTER_TAILSCALE_IP}:29500",
    rank=RANK, world_size=WORLD)

t = torch.ones(25_000_000)                      # 100MB fp32
dist.all_reduce(t)                              # warmup
t0 = time.time(); dist.all_reduce(t); dt = time.time() - t0
print(f"100MB all-reduce: {dt:.2f}s → {100/dt:.1f} MB/s")
```

- **이 `dt`가 라우터의 H 공식에 그대로 들어간다.** 모델 크기에 맞춰 환산한 `T_sync`를 Ji에게 넘길 것.
- NCCL은 시도하지 않는다.

**완료 기준:** 2머신 all-reduce 성공 + `T_sync` 수치 확보.

### N-3 · 워커 데몬 (2h)

`worker/daemon.py` — 각 GPU 머신에서 도는 프로세스.

```python
def collect_spec():
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,
        "vram_gb": round(p.total_memory / 1e9, 1),
        "cc": float(f"{p.major}.{p.minor}"),
        "vendor": "amd" if torch.version.hip else "nvidia",
        "chip_class": classify(p),           # 계약 5 — 라우터가 이걸로 디스패치
    }
```

- 시작 시 `worker.register`(`owner`, `share` 규칙, `credits_per_hour` 포함), 이후 1초마다 `worker.heartbeat`
- gpu_util: NVIDIA는 `pynvml.nvmlDeviceGetUtilizationRates`, AMD는 `rocm-smi --showuse`
- 연결 끊기면 지수 백오프 재접속
- `job.start` 수신 시 **Ethan의 트레이너를 서브프로세스로 띄우고** stdout 이벤트를 중계

**완료 기준:** 워커 2대가 등록·하트비트. 하나 죽이면 서버가 5초 안에 `worker.left`를 낸다.

### S-1 · FastAPI + WebSocket 허브 (1.5h)

```
WS   /ws/worker             워커 접속
WS   /ws/ui                 대시보드 접속 (팬아웃)
GET  /workers               레지스트리 (공유 상태, 크레딧 단가 포함)
POST /jobs                  잡 생성 (라우터의 설정 JSON)
POST /jobs/{id}/migrate     마이그레이션 트리거
PUT  /ckpt/{job}/{round}    체크포인트 업로드
GET  /ckpt/{job}/{round}    체크포인트 다운로드
GET  /credits/{owner}       잔액
```

- 레지스트리는 메모리 dict, 이벤트는 그대로 `/ws/ui`로 팬아웃
- **하트비트 감시자:** 5초 무응답 → `worker.left` (reason=timeout)
- 모든 이벤트를 S-8의 SQLite에 미러링

**완료 기준:** 워커 이벤트가 대시보드까지 끊김 없이 흐른다. Ji가 목 데이터를 끌 수 있다.

### N-4 · 체크포인트 전송 (2h)

- `PUT /ckpt/{job_id}/{round}` multipart + sha256 헤더, `GET`으로 수신
- 진행률을 `migration.progress`(phase=`transferring`, pct)로 스트리밍
- **이 진행바가 데모 비트 5에 나온다.** 조용히 성공하는 것보다 "진행 중"이 잘 읽힌다

**완료 기준:** 500MB 왕복 성공, 소요 시간 로깅.

### N-5 · 워커 회수/재합류 CLI (1h) ★무대에서 치는 명령

```bash
$ python -m worker.cli stop --reason game
  ✓ w1 (RTX 3090, junho) 주인이 회수 — 남은 워커로 라운드 계속
  ✓ junho에게 1.4 GPU-시간 적립  ·  잔액 128.5

$ python -m worker.cli start
  ✓ w1 재합류 — 다음 동기화(round 12)부터 참여
```

**출력이 깔끔해야 한다.** 심사위원이 터미널을 본다. 스택트레이스나 경고 금지. **크레딧 줄에 주목** — 이탈 데모가 공짜로 마켓플레이스 데모가 된다.

### N-6a · 공유 규칙 + 크레딧 원장 (0.75h) ★P0로 승격

마켓플레이스가 제품의 절반이므로 P1에 둘 수 없다.

```yaml
# ~/.gpushare/config.yaml
owner: junho
share:
  enabled: true
  not_gaming: true
  always_between: ["00:00", "08:00"]
credits_per_hour: 1.0
```

- 규칙을 `worker.register`에 실어 보낸다. 라우터가 뭘 빌릴지 고를 때 읽는다
- 원장: 소유자별 기여 GPU-시간 누적, `credits.update` 발행, `GET /credits/{owner}` 제공
- 수동 회수(N-5)만으로 데모가 된다. **자동 감지는 N-6b이고 필수가 아니다.**

**완료 기준:** 워커가 기여할 때 대시보드의 크레딧 잔액이 오르고, 주인이 빌릴 때 내려간다.

### N-6b · 자동 게임 감지 (1.25h, P1)

우리 트레이너가 아닌 PID가 GPU를 점유 → 자동 `stop --reason game`. 있으면 좋다. 수동 경로도 데모는 동일하다.

### S-8 · SQLite 실행 로깅 (1h)

```sql
workers(worker_id, owner, gpu, vendor, vram_gb, cc, chip_class, first_seen)
jobs(job_id, config_json, started_at, ended_at)
steps(job_id, worker_id, step, loss, step_time_s, tokens, ts)
syncs(job_id, round, participants_json, bytes, duration_s, global_loss, ts)
migrations(job_id, from_worker, to_worker, round, duration_s, ts)
probes(job_id, chip_class, config_name, t_step_median_s, tokens_per_s, peak_vram_gb, gpu_util)
credits(owner, delta_gpu_hours, reason, ts)
```

**`probes.chip_class`가 해자다.** 이 컬럼이 "3090 에이전트가 3090 데이터로 학습한다"를 슬로건이 아니라 실제로 만든다. 계약 5와 같은 문자열.

**발표에서 이 테이블들을 보여준다.** 비어 있으면 비전이 공허해진다.

## Jack의 의존성

| | |
|---|---|
| 기다리는 것 | Hour-0 계약만 |
| 나를 기다리는 사람 | **전원.** N-2가 Ethan의 방식을 결정하고, S-1 없이는 Ji가 목 데이터를 못 벗어난다 |
| T+8 이후 | 내 일은 대부분 끝. **Ethan을 돕는다** |

---
---

# 파트 2 — ETHAN: 훈련 실행 엔진

> "받은 설정대로 실행하고 측정값을 보고한다. 무엇을 할지는 정하지 않는다."

Ethan이 데모의 가장 강한 두 장면을 쥐고 있다. T-3(워커가 회수돼도 살아남음)과 T-5(마이그레이션). **일정이 밀리면 다른 걸 다 자르고 이 둘을 지킨다.**

## 유저 스토리

민지가 "훈련 시작"을 누르면 **loss가 내려가기 시작한다.** 빌린 GPU 두 대가 각자 학습하다 주기적으로 합쳐지고, 대시보드가 동기화 때마다 깜빡인다.

한 시간 뒤 친구 A가 집에 와서 발로란트를 켠다. 3090이 회수된다. **loss 곡선은 끊기지 않는다.** 민지는 알림을 보고서야 안다.

나중에 에이전트가 4090으로 옮긴다. 차트에 세로선이 그어지고 **곡선은 그대로 이어진다.** 민지가 아는 건 "옮겼는데 훈련이 안 망가졌다"뿐이다.

## 핵심 결정

| 항목 | 선택 | 비고 |
|---|---|---|
| 프레임워크 | PyTorch (CUDA + ROCm) | **같은 코드가 양 벤더에서 이미 돈다. 우리는 코드가 아니라 가중치를 옮긴다** |
| DiLoCo | torchft(1순위) / diloco_simple(폴백) | torchft의 DiLoCo 경로는 **experimental·nightly 전용**. 버전 핀 필수 |
| 백엔드 | gloo | NCCL 금지 |
| 체크포인트 | safetensors | 벤더 중립. **가중치만** |
| 모델 | nanoGPT급 decoder-only (~124M) | 24GB 카드 두 장은 48GB 한 장이 아니다. 메모리 풀링 없음 |

### DiLoCo 구조

```
각 워커:   θ_before = θ.clone()
          for _ in range(H):  inner AdamW 스텝
          pseudo_grad = θ_before − θ_after        ← 이걸 보낸다
전체:      outer optimizer (SGD + Nesterov) 가 pseudo_grad 평균을 적용
          → 새 global θ 를 모든 워커에 배포
```

**핵심 성질:** 동기화 직후엔 모든 워커가 동일한 global θ를 들고 있다. **그 순간이 마이그레이션의 유일한 안전 지점이고**, 그래서 inner optimizer 상태를 절대 옮기지 않는다.

## 티켓

### T-2a · torchft 판정 (1h) ★제일 먼저, 순서 무시

nightly 설치 → DiLoCo/LocalSGD 예제 실행 → **`requirements.txt`에 버전 핀**(experimental은 하룻밤에 깨진다).

**완료 기준:** 된다/안 된다가 팀 전체에 명확히 공유됨. **T+3까지.**

### T-1 · 단일 GPU 학습 루프 (1.5h)

`trainer/loop.py` — 계약 3 설정을 그대로 소비.

```python
def build(cfg):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg["dtype"]]
    model = NanoGPT(cfg["model"]).to("cuda", dtype=dtype)
    if cfg["attention"] == "sdpa":
        model.use_sdpa = True                     # F.scaled_dot_product_attention
    scaler = torch.amp.GradScaler() if cfg["dtype"] == "fp16" else None
    return model, scaler

# 고정 작업량 불변식
assert cfg["micro_batch"] * cfg["grad_accum"] * len(cfg["workers"]) * SEQ_LEN \
       == cfg["global_batch_tokens"]
```

- `train.step`을 stdout에 JSON 한 줄로 발행
- **이 assert를 꼭 유지할 것.** 칩 에이전트가 배치를 키웠을 때 이게 깨지면 우리는 *일을 덜 한* 것이지 빨라진 게 아니고, before/after 전체가 질문 한 번에 무너진다

### T-2 · DiLoCo 2워커 (3h) ★핵심

- inner AdamW H스텝 → pseudo-grad → outer SGD(Nesterov) → 배포
- `train.sync`에 `participants`, `bytes`, `duration_s`, `global_loss` 발행
- `bytes`는 실제로 센다 — T-8이 쓴다

**완료 기준:** 2머신에서 loss 하락, 동기화가 정확히 H스텝마다, 전송 바이트 로깅.

### T-3 · barrier 타임아웃 + 생존 워커 평균 (2h, 폴백 시 5h) ★★

**동기화 barrier에 타임아웃을 걸고 응답한 워커만으로 평균낸다.** 사라진 워커를 기다리며 블록하지 않는다. 그게 전부다.

- **torchft 경로:** lighthouse quorum 배선. 하트비트 health check가 이미 있다.
- **폴백 — 서버 중계 평균 (추천):**
  ```
  워커 → pseudo_grad를 서버에 POST (데드라인 포함)
  서버 → 데드라인까지 도착한 것만 평균 → 새 global θ 브로드캐스트
  ```
  집합 통신 대신 별 모양 토폴로지. **fault tolerance가 공짜로 따라온다** — 서버가 안 기다리면 그게 곧 내성이다. 워커 2~4대에 H가 크면 대역폭도 문제없고, gloo 의존까지 사라진다.
  → torchft가 안 되면 이 길로. dist collective 위에 직접 짜려 들지 말 것.

**완료 기준:** 주인이 훈련 중 GPU를 회수해도 훈련이 계속되고, 다음 sync의 `participants`가 줄어든 채로 찍힌다.
**이게 데모 비트 4 전체이고 T-5의 절반이다.**

### T-4 · 체크포인트 save/load (1h)

```python
from safetensors.torch import save_file
save_file(model.state_dict(), f"ckpt/{job}/round_{n}.safetensors")
json.dump({"round": n, "global_step": s, "global_loss": gl,
           "config_hash": h, "ts": time.time()},
          open(f"ckpt/{job}/round_{n}.meta.json", "w"))
```

**가중치만.** inner optimizer 상태는 저장하지 않는다 — 동기화 경계에서만 옮기므로 필요 없다.

### T-5 · 마이그레이션 실행 (1.5h) ★★

```
1. 다음 동기화 대기          migration.progress phase=waiting_sync
2. 저장                      phase=saving
3. 업로드 (Jack의 N-4)       phase=transferring  pct=...
4. 대상 워커 로드            phase=loading
5. 재개                      phase=resumed
```

**구현상 마이그레이션 = T-3(워커 이탈) + 신규 워커 합류다.** 새 메커니즘이 아니다. 1단계가 전부이고 나머지는 파일 이동이다.

**대기를 화면에 드러낼 것.** 유저가 마이그레이션을 누르면 다음 동기화까지 수 초~수 분 기다린다. 숨기지 말고 `waiting for sync`를 보여준다 — **왜 기다리는지가 곧 왜 안전한지의 설명**이다.

**완료 기준:** **NVIDIA→NVIDIA 먼저 성공.** `global_loss`가 마이그레이션을 건너 이어진다. ROCm이 되면 대상만 AMD로 교체 — 메커니즘이 동일하므로 ROCm이 끝내 안 돼도 데모는 온전하고 "벤더 간"이라는 수식어만 잃는다.

### T-6 · 프로브 러너 (1.5h)

```python
def probe(cfg, steps=40, warmup=10):
    ts = [run_one_step(cfg) for _ in range(steps)]
    t_step = statistics.median(ts[warmup:])          # 평균 아님, 중앙값
    return {"t_step_median_s": t_step,
            "tokens_per_s": cfg["global_batch_tokens"] / t_step,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu_util": sample_util(),
            "chip_class": CHIP_CLASS}                # 계약 5

def search_batch(cfg, target=0.85):
    mb = cfg["micro_batch"]
    while True:
        try:
            probe({**cfg, "micro_batch": mb * 2}, steps=3, warmup=1)
            if peak_vram_frac() > target: return mb
            mb *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); return mb
```

**앞 10스텝은 반드시 버린다**(cudnn autotune, allocator warmup, compile). baseline에 포함시키면 baseline을 인위적으로 느리게 만든 것이고, 팀이 스스로를 속이는 가장 흔한 방식이다.

**모든 프로브에 `chip_class`를 태깅**해서 올바른 칩 에이전트의 이력에 쌓이게 한다.

### T-7 · global loss 평가 (0.5h)

**고정된 held-out 배치**로 동기화마다 평가해 `train.sync.global_loss`에 싣는다.

**마이그레이션이 훈련을 보존했다는 유일한 증거다.** 스텝별 loss는 마이그레이션 직후 약간 튄다(inner optimizer 모멘트 재구축). global loss는 **정의상 연속**이다 — 옮긴 게 바로 그 global θ이므로.

### T-8 · 통신량 + DDP 베이스라인 (1h, P1)

같은 모델·스텝 수로 DDP를 한 번 돌려 바이트 측정. **헤드라인은 비율이다: `DDP 12.4GB → 우리 40MB (310×)`.** 절대량만으로는 듣는 사람에게 의미가 없다. 사전 작업이고 라이브가 아니다.

## Ethan의 의존성

| | |
|---|---|
| 기다리는 것 | N-2(gloo) — 단 T-1은 즉시 시작 가능 |
| 나를 기다리는 사람 | Ji, 단 목 데이터로 T+8까지 버틴다 |
| 위험 | T-2a 실패 시 T-3이 2h → 5h. **그때 Jack이 붙는다** |

---
---

# 파트 3 — JI: 에이전트 + 최적화 + 대시보드

> "차별점이고, 코드상으로 사실이어야 한다."

렌탈 결정, 스케줄링, 칩 선택, 설정 최적화, 마이그레이션이 **라우터와 칩 에이전트 두 단**으로 전부 여기 있다.

대시보드도 Ji가 갖는다. 심사위원이 실제로 보는 건 "에이전트가 똑똑한 걸 했다"는 화면 한 장이라, 규칙 쓰는 사람과 화면 그리는 사람이 같아야 한다.

## 유저 스토리

민지는 bf16이 뭔지 모르고 알 필요도 없다. 훈련 시작 전에 에이전트가 먼저 말을 건다.

```
Ampere24GB 에이전트가 3090들을 30초 재봤습니다.
  현재 설정 그대로:  5시간 08분  ·  38 크레딧
  제 설정으로:       3시간 42분  ·  28 크레딧
  바꾼 것: fp32→bf16 · 배치 8→32(누적 4→1) · sdpa · H=190
  optimizer step당 토큰: 32,768 → 32,768  ✓
                                  [이대로] [기본값으로]
```

나중에:

```
💡 라우터: AMD 7900 유휴 · 시간당 42% 저렴
   → 마이그레이션?               [수락] [거절]
```

**Ji가 만드는 건 민지가 끝까지 배울 필요 없는 모든 것이다.** NCCL이 왜 안 되는지, H가 왜 190인지, 3090이 4090과 뭐가 다른지. 그 결정들이 민지 *대신* 내려졌다는 게 제품이다.

**Ji가 무대에서 답해야 할 질문:** *"그냥 스케줄러 아닌가요?"*
→ "스케줄러는 배치 시점에 빈 슬롯을 채웁니다. 저희는 뭘 빌릴지, 각 칩 종류가 어떤 설정을 원하는지, 언제 옮길지를 **실행 중 측정값으로** 고릅니다. 그리고 그 측정값이 칩 종류별로 쌓입니다."

## 티켓

| ID | 티켓 | 시간 | P |
|---|---|---|---|
| S-2 | 목 이벤트 발생기 | 0.5h | **P0 ★제일 먼저** |
| S-3 | 대시보드 기본 | 2.5h | P0 |
| S-4 | **라우터 + 칩 에이전트 구조** | 2h | P0 |
| S-5 | H 계산 + 낙오자 판정 | 1h | P0 |
| S-6 | 최적화 전/후 화면 | 1.5h | P0 |
| S-7 | 마이그레이션 제안 카드 + 시각화 | 1.5h | P0 |
| **S-9** | **크레딧 가격 + 렌탈 결정** | **1.5h** | **P0** ← P1이었음 |

### S-2 · 목 이벤트 발생기 (0.5h) ★★제일 먼저

계약 1 이벤트를 가짜로 쏘는 스크립트. 30초 시나리오: 워커 2대 학습 → 주인이 하나 회수 → 마이그레이션 → 크레딧 적립.

**30분짜리인데 보드에서 가장 남는 티켓이다.** Ji가 T+2부터 T+20까지 한 번도 안 막힌다. 없으면 훈련이 돌 때까지 놀다가 후반에 대시보드를 급조하게 되는데, 대시보드는 데모의 유일한 출력 장치다.

### S-3 · 대시보드 기본 (2.5h)

```
┌──────────────────────────────────────────────────┐
│  훈련: small-llm-v2                  상태: ● 진행 │
│  빌린 칩:                                         │
│   • friendA 3090  [████████] 활발  1.0 cr/h      │
│   • friendB 3090  [████████] 활발  1.0 cr/h      │
│   • lab AMD7900   [가용]           0.6 cr/h      │
│  Loss: 2.4 → 1.1 ↓                                │
│   ─ global loss (굵게, 동기화마다 한 점)          │
│   ┄ step loss   (얇은 회색)                       │
│  동기화: H=190마다 ●    전송: 40MB/라운드         │
│  소비: 12.4 cr    ·    AWS 환산: $6.20            │
└──────────────────────────────────────────────────┘
```

- 워커 회수 → 전환 효과와 함께 회색
- 마이그레이션 → 세로선 + 전후 값
- 공급자 뷰: 크레딧 잔액 증가

### S-4 · 라우터 + 칩 에이전트 구조 (2h) ★구조는 진짜로

**P0에서 규칙은 얇다. 구조는 얇으면 안 된다.** 심사위원이 이 파일을 열었을 때 if문 세 개짜리 함수 하나가 있으면 "칩마다 전용 에이전트"가 부채가 된다.

```python
class ChipAgent:
    """칩 종류마다 하나. 지금은 규칙, 나중엔 학습 모델 — 인터페이스는 같다."""
    chip_class: str
    def decide(self, job, probe) -> dict: ...

class Ampere24GB(ChipAgent):        # RTX 3090
    chip_class = "ampere_24gb"
    def decide(self, job, probe):
        return {"dtype": "bf16", "attention": "sdpa",
                "micro_batch": probe.searched_batch}

class Ada24GB(ChipAgent):           # RTX 4090
    chip_class = "ada_24gb"
    ...

class DefaultAgent(ChipAgent):      # 모르는 칩 — 콜드 스타트
    chip_class = "default"
    def decide(self, job, probe):
        return {"dtype": "bf16" if probe.cc >= 8.0 else
                         "fp16" if probe.cc >= 7.0 else "fp32",
                "attention": "sdpa", "micro_batch": probe.searched_batch}

class Router:
    def agent_for(self, worker) -> ChipAgent:
        return REGISTRY.get(worker.chip_class, DefaultAgent())

    def plan(self, job, workers, probes):
        per_chip = {w.id: self.agent_for(w).decide(job, probes[w.id])
                    for w in workers if w.trainable}
        cfg = reconcile(per_chip)                            # 풀 전체 조정
        cfg["H"] = compute_H(probes.t_sync, probes.t_step)   # 풀 단위, S-5
        cfg["grad_accum"] = job.global_batch_tokens // (
            cfg["micro_batch"] * len(workers) * SEQ_LEN)     # 고정 작업량 불변식
        return cfg
```

두 가지:

- **`worker.chip_class`는 계약 5이고, Jack이 SQLite에 쓰는 것과 같은 문자열이다.** 이게 칩별 학습을 슬로건이 아닌 실제로 만든다.
- **`reconcile()`이 중요하다:** 워커끼리 의견이 다를 수 있다(3090과 4090은 다른 배치를 원한다). P0에선 풀 전체에서 가장 보수적인 dtype을 택하고 배치는 워커별로 둔다.

**레버 순서가 중요하다.** 메모리 절약 레버 먼저(bf16 → sdpa → checkpointing), **그다음에** 배치 탐색. 반대로 하면 bf16 켠 뒤 배치를 다시 재야 한다. 전체 표는 부록 A.

### S-5 · H 계산 + 낙오자 판정 (1h) ★라우터의 간판

```python
def compute_H(t_sync, t_step, rho=0.05):
    h = int(t_sync / t_step * (1 - rho) / rho)
    return max(50, min(500, h))                    # clamp
```

| 상황 | T_step | T_sync | H | 비고 |
|---|---|---|---|---|
| Tailscale direct | 0.20s | 2s | 190 | 정상 |
| DERP 릴레이 | 0.20s | 25s | 500 | **상한 도달 → 경고** |
| 같은 LAN 유선 | 0.20s | 0.5s | 50 | 하한 |

**이름을 걸 레버가 이것이다.** bf16이나 배치 튜닝은 어디서나 쓰는 표준이라 "에이전트가 진짜 한 거냐"에 약하다. H는 다르다 — **측정된 네트워크 상태에 반응해 값이 달라지는 유일한 레버**이고, 사람이 굳이 계산하지 않을 종류다. Jack이 "릴레이로 떨어졌다"고 하면 H가 자동으로 올라가는 게 데모 장면이 된다.

H는 풀 전체 단위라 칩 에이전트가 아니라 라우터의 것이다.

- 상한 도달 시: `"네트워크가 느려 H 상한 도달. 모델 축소 권장"`
- 데모에선 ρ를 0.2로 올려 5분 안에 동기화 2회가 보이게. **그렇게 했다고 말할 것.**

```python
def check_stragglers(workers):
    med = statistics.median(w.t_step for w in workers if w.trainable)
    for w in workers:
        if w.t_step > 3 * med:      reassign(w, "preprocess")
        elif w.t_step > 1.2 * med:  w.micro_batch = int(w.micro_batch * med / w.t_step)
    # ⚠️ 배치가 다르면 outer average를 샘플 수로 가중해야 한다. Ethan과 합의.
```

### S-6 · 최적화 전/후 화면 (1.5h)

`probe.result` 두 개로 렌더링. **세 줄이 방어선이다:**

- `optimizer step당 토큰: 32,768 = 32,768 ✓` — 같은 양의 일
- `측정: 30스텝(warmup 10 제외) → 투영값` — 5시간을 잰 척하지 않는다
- `loss 곡선 겹침 ●` — 빨라진 대신 나빠진 게 아니다

**절약된 크레딧을 절약된 시간 옆에 표시.** 같은 화면에서 두 층이 이어진다.

**baseline 주의.** fp32에 배치 1이면 숫자는 화려하지만 "아무도 그렇게 안 씁니다"에 죽는다. **PyTorch 기본값 + 초보자가 흔히 쓰는 설정**으로 잡고, 숫자를 보여주기 전에 baseline이 뭐였는지 먼저 밝힌다. 전체 프로토콜은 부록 A.

### S-7 · 마이그레이션 제안 카드 + 시각화 (1.5h)

- 규칙: 대상 가용 && (더 빠름 || 크레딧 더 쌈) && 기대 이득 > 마이그레이션 비용
- 수락 → `POST /jobs/{id}/migrate` → `migration.progress` 5단계 렌더링, **`waiting for sync` 포함**
- 완료 시 세로선 + `이전 1.13 → 재개 1.13`

**수락 버튼을 유지한다.** 완전 자동은 P2이고 그렇게 말하는 게 주장하는 것보다 낫다. 에이전트가 결정하고 유저가 통제를 갖는 구조이고, 보이는 클릭이 조용히 일어나는 것보다 훨씬 잘 시연된다.

### S-9 · 크레딧 가격 + 렌탈 결정 (1.5h) ★P0로 승격

라우터의 첫 결정이 *뭘 빌릴까*라서 이건 표시 기능이 아니다.

```python
def pick_chips(job, market):
    """실제로 충분한 것 중 제일 싼 칩 — 구세대 포함."""
    fits = [c for c in market if c.vram_gb >= job.min_vram
                              and c.cc >= 7.0 and c.available]
    return sorted(fits, key=lambda c: c.credits_per_hour)[:job.want]
```

표시:
```
GPU-시간 0.42 × 피어 단가 1.0 cr/h  = 12.4 크레딧
GPU-시간 0.42 × AWS 온디맨드 $X/h   = $6.20 환산
```

**계산식을 숫자 옆에 띄운다.** 안 그러면 Q&A 첫 질문이 "그거 어떻게 나왔죠?"다.

발표 대사: *"이 작업은 4090이 필요 없습니다. 라우터가 충분하기만 한 칩을 훨씬 낮은 단가로 찾았습니다."*

## Ji의 의존성

| | |
|---|---|
| 기다리는 것 | Hour-0 계약만. **S-2 덕에 이후 아무도 안 기다린다** |
| 나를 기다리는 사람 | Ethan이 라우터의 설정 JSON을 소비 — 형식을 먼저 고정할 것 |
| 합의 필요 | `chip_class` 문자열(계약 5)은 Jack과 · 샘플 수 가중은 Ethan과 |

---
---

# 파트 4 — 통합

혼자 오래 가면 안 붙는다. **정해진 시각에 강제로 합친다.**

| 시각 | 합치는 것 | 성공 기준 |
|---|---|---|
| **T+3** | Jack N-2 · Ethan T-2a | gloo 여부, torchft 여부. **필요하면 여기서 계획 수정** |
| **T+8** | Jack N-3, S-1 + Ethan T-2 + Ji S-3 | 진짜 훈련 이벤트가 진짜 대시보드에. **목 데이터 졸업** |
| **T+13** | Ethan T-3 + Jack N-5, N-6a + Ji S-3 | 주인 회수가 화면에, 크레딧 적립 (비트 4 완성) |
| **T+17** | Ethan T-5 + Jack N-4 + Ji S-7 | 마이그레이션 전 과정 가시, global loss 연속 (비트 5 완성) |
| **T+19** | 전원 | 데모 3회 연속 성공. **코드 동결.** 리허설과 백업 영상만 |

**T+19 이후 기능 추가 없음.**

## 여러 사람에 걸친 것 — 주인을 명시

**마이그레이션:**

| 부분 | 티켓 | 담당 |
|---|---|---|
| 전송 | N-4 | Jack |
| 실행 (대기·저장·로드·재개) | T-5 | Ethan |
| 판단·제안 카드·시각화 | S-7 | Ji |

데모의 클라이맥스가 셋에 흩어져 있다. **T+17에 셋이 같이 앉는다.**

**`chip_class`(계약 5):** Jack이 쓰고(N-3, S-8), Ethan이 태깅하고(T-6), Ji가 디스패치한다(S-4). 셋이 다르면 칩별 학습이 조용히 안 된다. **Hour 0에 합의할 것.**

## 부하 불균형

Jack의 일은 **앞에 몰려 있고**(T+8이면 대부분 끝), Ethan은 **뒤에 몰려 있다.** 게다가 torchft 실패 시 T-3이 2h → 5h로 불어난다.
→ **T+8 이후 Jack이 Ethan을 돕는다.** 예외가 아니라 계획이다.

## 자를 순서

위에서부터 자른다. 선 아래는 자르면 데모가 무너진다.

1. N-6b 자동 게임 감지 → **수동 CLI로. 데모는 동일하다**
2. T-8 DDP 베이스라인 → 절대 바이트만 표시
3. S-6 최적화 화면 → 에이전트 판단을 로그로만
4. 칩 에이전트 2종 초과분 → `Ampere24GB`와 `DefaultAgent`만 유지, 구조 증명엔 충분
5. AMD 마이그레이션 → **NVIDIA→NVIDIA로.** 메커니즘 동일, 수식어만 잃음
6. ─────── 여기부터 자르면 안 됨 ───────
7. S-9 렌탈 결정 + 크레딧 *(제품의 절반)*
8. N-6a 공유 규칙 + 원장 *(제품의 절반)*
9. T-5 마이그레이션
10. T-3 이탈 내성
11. T-2 DiLoCo

> 바뀐 것에 주목: **크레딧과 렌탈이 선 아래로 내려왔다.** 자르면 마켓플레이스 절반이 사라지고 포지셔닝이 성립하지 않는다.

---
---

# 파트 5 — 데모

5분. **한 문장을 심어야 한다:**

> "GPU가 없었다. 친구 셋에게서 빌렸고, 에이전트가 튜닝하고 묶었고, 주인이 하나 회수해 갔는데도 훈련이 멈추지 않았다."

나머지는 전부 증거이고, **증거는 loss 곡선이다.**

## 발표 전

### 현장 도착 (T-30) — 네트워크부터, 다른 건 나중

| 확인 | 방법 | 실패 시 |
|---|---|---|
| Tailscale `direct`인지 DERP `relay`인지 | `tailscale status` | 릴레이면 H 올리고(50→200) 모델 축소. 그 자리에서 판단 |
| gloo all-reduce 동작 | 사전 작성한 테스트 스크립트 | 핫스팟 → 유선 → 로컬 2프로세스 |
| `GLOO_SOCKET_IFNAME=tailscale0` | 각 워커 환경변수 | — |

### 무대 직전 (T-10)

| 항목 | 담당 |
|---|---|
| **훈련이 이미 돌고 있음** (수백 스텝 진행) | Ethan |
| 워커 A, B 접속, 크레딧 적립 중 | Jack |
| 마이그레이션 대상 칩 **워밍업 완료** | Ethan |
| 고정 held-out 평가 배치 확인 | Ethan |
| 대시보드 전체화면, 배율 조정 | Ji |
| 백업 영상 탭 (숨김) | Ji |

**무대에서 아무것도 시작하지 않는다.** 초기화도, 모델 로드도, 최초 접속도. 라이브로 치는 건 **회수 하나, 마이그레이션 수락 하나**뿐이다.

행사장 와이파이가 초기 핸드셰이크를 죽이는 게 데모를 망치는 1번 원인이다. 이미 돌던 잡에 붙는 건 실패해도 티가 안 나지만, 무대에서 시작하는 건 아니다.

## 비트

### 1 — 훅 (0:00–0:30)
노트북을 들며. *"이 맥북엔 GPU가 없습니다. 그런데 지금 모델이 훈련되고 있습니다 — 자기 것이 아닌 GPU 세 대 위에서."* 대시보드 전환: loss 하강, 빌린 워커 둘, 크레딧 적립 중.
아키텍처도, 스택도, 팀 소개도 없다. 30초 안에 화면이 뜬다.

### 2 — 두 겹 (0:30–1:15)
*"GPU는 없는 게 아니라 흩어져서 놀고 있습니다. Vast.ai는 데이터센터를 팔지만 묶는 건 여전히 여러분 몫이고, Ray는 묶어주지만 칩은 여러분이 찾아야 합니다. 우리는 둘 다 합니다 — **게다가 뭘 빌릴지까지 에이전트가 고릅니다.**"*
크레딧 단가가 붙은 대여 메뉴를 보여준다. 세 번째가 **다른 벤더**임을 심어둔다 — 비트 5의 복선.

### 3 — 묶이고 동기화 (1:15–2:15)
*"각 칩이 따로 학습하다 190스텝마다 한 번 동기화합니다. 통신량이 수백 배 줄어서 데이터센터 패브릭이 아니라 집 와이파이로도 됩니다."*
동기화가 깜빡일 때 짚는다. **헤드라인은 비율:** `DDP였다면 12.4GB · 우리 40MB (310×)`.

### 4 — 주인이 귀가 (2:15–3:00) ★가장 강한 장면
*"친구가 집에 와서 발로란트를 켭니다."* **터미널에서 실제로 워커를 회수한다.**
워커 하나가 회색이 되고, 주인의 크레딧 잔액이 뛴다. loss 곡선은 계속 내려간다.
*"그의 GPU는 즉시 돌아갔고, 세 시간치 크레딧을 받았고, 그녀의 훈련은 안 멈췄습니다. 양쪽 다 원하는 걸 얻었습니다."*

**⚠️ 표현 주의:** "DiLoCo가 이탈에 강건하다"고 말하지 말 것. DiLoCo가 주는 건 싼 통신이지 복구가 아니다. 논문을 아는 사람이면 바로 파고든다:
> *"DiLoCo는 이탈을 싸게 만들 뿐이고, 복구 로직은 저희가 그 위에 얹었습니다."*

**보조 워커를 회수할 것.** 비트 5에 쓸 칩은 절대 건드리지 않는다.

### 5 — 에이전트의 결정 (3:00–4:00) ★차별점이 서는 지점

```
💡 라우터: AMD 7900 유휴 · 시간당 42% 저렴
   → 마이그레이션?   [수락] [거절]
```

*"여기부터가 저희가 만든 부분입니다. 두 단입니다 — 뭘 빌리고 언제 옮길지 정하는 라우터, 그리고 그 칩이 어떤 설정을 원하는지 아는 칩 종류별 에이전트."*

[수락] 클릭. 단계가 보인다: `동기화 대기 → 저장 → 전송 → AMD에서 로드 → 재개`.

첫 단계를 짚으며: *"기다립니다. 동기화 시점엔 모든 워커가 같은 가중치를 들고 있어서, 그 외엔 옮길 게 없습니다."*

그리고 결정타:
> *"loss가 이어집니다. NVIDIA에서 저장한 게 AMD에서 살아났습니다. **코드를 변환하는 게 아닙니다 — 같은 PyTorch가 양쪽에서 돕니다. 저희는 벤더 중립 가중치 파일 하나를 옮겼습니다.**"*

**숫자를 명시적으로 띄운다:** `이전 1.13 → 재개 1.13`. 곡선만으로는 이어진 것처럼 그린 것과 구분되지 않는다.

이 1분을 서두르지 말 것. 발표 전체다.

### 6 — 양쪽 장부 (4:00–4:30)
*"이 작업은 12.4 크레딧. AWS면 $6.20."* **계산식을 화면에.**
*"그리고 4090은 필요 없었습니다 — 라우터가 충분하기만 한 칩을 골랐습니다."*
그리고 뒤집어서: *"방금 빠진 3090의 주인 준호는 수업 간 사이 크레딧을 벌었습니다. 다음 달 자기 프로젝트에 씁니다. 놀 때 빌려주고 필요할 때 빌려 씁니다."*

### 7 — 비전과 마무리 (4:30–5:00)
*"오늘은 사람이 수락을 눌렀습니다. 다음은 라우터가 혼자 옮깁니다. 그리고 기록되는 모든 실행이 그 칩의 에이전트로 들어갑니다 — 3090 에이전트는 3090 데이터로만 똑똑해집니다. 그게 저희 해자입니다."*
> *"훈련에 데이터센터는 필요 없습니다. 칩은 이미 존재합니다 — 그냥 남의 방에서 놀고 있을 뿐입니다."*

## 무대 위 역할

| | 역할 | 하는 일 |
|---|---|---|
| **Ji** | 발표자 | 전체 내레이션. 대시보드 조작. 수락 클릭 |
| **Ethan** | 훈련 | 사전에 잡 시작. 비트 4에서 회수 명령. 기술 Q&A |
| **Jack** | 네트워크 | 접속 감시. 끊기면 **말없이** 복구. 폴백 판단 |

**발표자는 절대 멈추지 않는다.** Jack이 조용히 고치고, 30초 안에 복구 안 되면 합의한 폴백으로.

## 폴백

| 터지는 것 | 폴백 | 발화 |
|---|---|---|
| 와이파이 죽음 | ① 핫스팟 → ② 유선 → ③ 한 머신 2프로세스 | "네트워크가 불안해서 로컬 두 프로세스로 같은 구조를 보여드리겠습니다" |
| Tailscale DERP 릴레이 | H 50→200, 모델 축소. **비트 3 타이밍 재계산** | (언급 안 함) |
| 워커 접속 실패 | 이미 돌던 잡으로 진행, 접속 장면 생략 | (언급 안 함) |
| **AMD 없음 / ROCm 안 됨** | **NVIDIA→NVIDIA로.** 메커니즘 동일, 데모 온전 | "여기선 같은 벤더 간 이동이지만 체크포인트가 벤더 중립이라 대상만 바꾸면 됩니다" |
| AMD 되는데 무대에서 실패 | 녹화 클립 + **체크포인트 파일 실물**을 터미널에서 | "이 부분은 사전 녹화입니다" — 들키는 것보다 낫다 |
| loss 발산 / NaN | 미리 구운 체크포인트에서 재개 | "체크포인트에서 복구하겠습니다" (기능 시연이 됨) |
| 대시보드 죽음 | 터미널 로그로 | 로그 출력을 미리 읽기 좋게 |
| 전부 죽음 | 3분 풀 녹화 | 최후 수단. **반드시 준비** |

**백업 영상은 옵션이 아니다.** 전날 찍는다.

## Q&A

**크레딧인가요 실제 돈인가요?** ★반드시 나옴
> 지금은 크레딧입니다. GPU-시간을 기여하고 나중에 씁니다. 정산·분쟁·KYC를 피할 수 있고, 빌려주기도 하고 빌리기도 하는 저희 타깃 유저에게 호혜 구조가 맞습니다. 현금 지급은 기술이 아니라 사업 판단입니다.

**왜 남에게 자기 GPU를 빌려주나요?**
> 어차피 놀고 있고 비용이 안 들기 때문입니다 — 주인은 즉시 우선권을 되찾습니다, 방금 보신 대로요. 그리고 저희 유저 대부분은 양쪽에 있습니다. 일하는 동안 빌려주고 규모가 필요할 때 빌립니다.

**NCCL이 NAT 넘어서 됩니까?** ★가능성 높음
> 안 됩니다. 그래서 gloo를 씁니다. NCCL은 스텝마다 통신을 전제하는데 저희는 H스텝에 한 번이라 CPU 측 collective로 충분합니다. **NCCL이 못 가는 네트워크를 쓸 수 있게 만드는 게 DiLoCo를 고른 이유입니다.** torchft도 replica group 사이엔 gloo를 씁니다.

**CUDA를 ROCm으로 트랜스파일하나요?**
> 아니요, 그럴 필요도 없습니다. 같은 PyTorch 코드가 양 벤더에서 이미 돕니다. 저희는 코드가 아니라 벤더 중립 가중치 파일을 옮깁니다.

**그냥 스케줄러 아닌가요? 쿠버네티스로 안 되나요?**
> 스케줄러는 배치 시점에 빈 슬롯을 채웁니다. 저희는 뭘 빌릴지, 칩 종류마다 어떤 설정을 쓸지, 언제 옮길지를 실행 중 측정값으로 정합니다. 그리고 그 측정값이 칩 종류별로 쌓입니다.

**Vast.ai랑 뭐가 다릅니까?**
> Vast.ai는 대여 마켓이고 빌린 뒤 묶는 건 여러분 몫입니다. 저희는 묶는 것까지 하고, 뭘 빌릴지도 에이전트가 고릅니다.

**Petals나 Hivemind랑은요?**
> 그쪽은 이미 가진 칩을 묶고, 추론 중심에 단일 스택입니다. 저희는 마켓플레이스 + 이종 벤더 오케스트레이션이고 훈련이 주 용도입니다.

**CUDA와 ROCm을 어떻게 같이 돌립니까?**
> 안 섞습니다. 그게 설계 규칙입니다. 같은 벤더는 동시에, 다른 벤더는 이동. 이 선을 그어서 어려운 부분을 피했습니다.

**옵티마이저 상태는 어떻게 옮깁니까?**
> 안 옮깁니다. 동기화 경계에서 옮기니 옮길 게 없습니다 — 그 순간엔 global 가중치가 공유 상태의 전부고, 새 워커는 신규 합류와 똑같이 inner loop를 시작합니다. inner loop 도중 이동은 별개 문제이고 다음 단계입니다.

**칩 에이전트라는 게 그냥 if문 아닙니까?**
> 지금은 규칙 맞습니다. 하지만 칩 종류마다 하나의 인터페이스 뒤에 자기 에이전트가 있고, 모든 실행이 그 에이전트의 이력으로 쌓입니다. 데이터가 충분해지면 주변을 안 바꾸고 클래스 하나씩 학습 모델로 교체합니다.

**torchft 갖다 쓴 거 아닌가요?**
> 훈련 코어는 검증된 걸 씁니다. 저희가 만든 건 마켓플레이스, 이종 풀 관리, 벤더 간 마이그레이션, 그리고 결정하는 에이전트입니다. DiLoCo를 바닥부터 다시 짜는 건 해커톤에서 할 일이 아닙니다.

**남의 GPU에 내 데이터를 왜 올립니까?**
> 아직 안 풀었습니다. 지금은 신뢰 기반이고 암호화·신뢰 점수·TEE가 로드맵에 있습니다. — **인정할 것.** 과장하다 걸리는 게 훨씬 비싸다.

**규모가 커져도 수렴합니까?**
> 보여드린 건 작은 모델입니다. 큰 규모는 검증 못 했고, 근거는 DiLoCo 문헌이 다루는 범위까지입니다.

**낙오자가 전체를 붙잡는 문제는요?**
> 라우터가 느린 칩을 훈련에서 빼고 데이터 전처리로 돌립니다. (구현했으면 시연, 아니면 "설계돼 있고 다음 단계입니다")

## 리허설 체크리스트

- [ ] 전체 시연 **3회 연속 성공**
- [ ] 각 폴백 경로를 최소 1회씩 실제로 실행
- [ ] 와이파이를 일부러 끊고 복구 시간 측정
- [ ] 대사에 "DiLoCo가 이탈에 강건" 류 표현이 없는지
- [ ] 대사에 "코드를 옮긴다" 류 표현이 없는지
- [ ] 크레딧 계산식과 AWS 환산이 화면에 보이는지
- [ ] 비트 4에서 크레딧 잔액이 눈에 띄게 움직이는지
- [ ] 마이그레이션 전후 loss 값이 숫자로 뜨는지
- [ ] 백업 영상 완성
- [ ] 배터리 / 충전기 / HDMI 어댑터 / 핫스팟 데이터

---
---

# 부록 A — 최적화 레버와 측정 (Ji)

## 모든 것이 기대는 불변식

> **최적화는 같은 일을 더 빨리 하는 것이다. 덜 하고 빨리가 아니다.**

배치를 두 배로 하면 스텝 수가 절반이 된다. 스텝/초는 좋아 보이지만 같은 양의 일을 한 게 아닐 수 있다. 그래서 **모든 측정은 tokens/sec(훈련) 또는 고정 길이 output tokens/sec(추론)으로 한다.** 스텝/초, 요청/초는 절대 안 쓴다.

| 불변식 | 훈련 | 추론 |
|---|---|---|
| 일의 양 | optimizer step당 토큰 고정 (배치↑는 grad_accum↓로 상쇄) | 요청 수·프롬프트·출력 토큰 고정 (`ignore_eos` + `max_tokens`) |
| 품질 | 동일 seed·데이터 순서 → loss 곡선 겹쳐보기 | 양자화 시 perplexity 또는 샘플 출력 비교 |

## 결정 파이프라인

```
Phase 0  정적 판독   (0초)      모델 스펙, chip_class, 버전 → 칩 에이전트로 라우팅
Phase 1  프로브      (30~60초)  20~50 실제 스텝 → step time, peak mem, util, sync time
Phase 2  규칙 적용   (즉시)     칩 에이전트가 설정 · 라우터가 H와 풀 결정
Phase 3  실행 중 감시 (상시)    낙오자 감지, H 재조정, 마이그레이션 판단
```

**Phase 1이 핵심이다.** 스펙만 보고 정하는 건 그냥 조회표고, "그거 하드코딩 아니냐"에 답이 없다. 재고 정한다는 게 차별점이다.

## 훈련 레버

| # | 레버 | 규칙 | 효과 | 리스크 | 담당 |
|---|---|---|---|---|---|
| 1 | bf16 / fp16 | `cc ≥ 8.0` → bf16; `7.0~8.0` → fp16 + GradScaler; `< 7.0` → 훈련 제외 | 1.5~2× | bf16은 거의 없음; fp16은 loss scaling | 칩 에이전트 |
| 2 | SDPA / Flash Attention | dtype이 fp16/bf16이고 head_dim 지원 | 메모리 대폭↓, 긴 seq 속도↑ | custom attention은 교체 필요 | 칩 에이전트 |
| 3 | 배치 크기 탐색 | peak VRAM **85%**까지 2배씩; OOM → 직전 값. **global batch는 grad_accum으로 고정** | util 60~95% 구간에서 최대 | OOM — try/except + 롤백 필수 | 칩 에이전트 |
| 4 | H (동기화 주기) | 아래 공식 | 통신 대기 제거 | 너무 크면 수렴 품질↓ | **라우터** |
| 5 | 낙오자 처리 | 아래 규칙 | 동기화 대기 제거 | 워커 수 감소 | **라우터** |
| 6 | dataloader 튜닝 | util < 90% **이면서** 데이터 대기 비중 큼 → `num_workers`↑, `pin_memory` | 데이터 병목일 때만 큼 | 없음 | 칩 에이전트 |
| 7 | torch.compile | `예상 실행시간 × 0.15 > 컴파일 비용(~60초)` | 1.1~1.4× | **짧은 작업엔 순손해** | 칩 에이전트 |
| 8 | activation checkpointing | 목표 배치에서 **OOM일 때만** | 메모리 30~40%↓ | 연산 ~30%↑; 메모리 병목 아니면 손해 | 칩 에이전트 |
| 9 | outer gradient 양자화 | 측정 대역폭이 임계 이하 → int8 pseudo-grad | 동기화 시간 대폭↓ | 구현 난이도 | 라우터 |

### 레버 순서

```
1. bf16               → 메모리 절반
2. SDPA               → 메모리 추가 절감
3. (필요시) checkpointing
4. 그다음에 배치 탐색   ← 1~3이 만든 여유를 현금화
5. 프로브로 step time
6. sync time 측정 → H 계산
7. 워커별 step time 비교 → 낙오자 처리
```

**메모리를 아끼는 레버가 먼저, 쓰는 레버가 나중.**

### 레버 4 — H 공식

```
overhead = T_sync / (H × T_step + T_sync) ≤ ρ
→  H ≥ (T_sync / T_step) × (1 − ρ) / ρ
```

ρ = 0.05면 `H ≥ 19 × T_sync / T_step`. **[50, 500]으로 clamp** — DiLoCo 문헌 범위 밖은 수렴을 주장할 수 없다. 상한에 걸리면 경고하고 모델 축소를 권한다.

### 레버 5 — 낙오자 규칙

라운드 시간은 가장 느린 워커가 정하므로, 낙오자 하나가 라운드당 `(T_slow − T_median) × H`를 낭비한다.

1. **먼저 배치 균형을 시도** — micro-batch를 step time에 반비례 배분해 step time을 맞춘다. 느린 칩도 기여한다.
   **⚠️ 그러면 outer average를 워커별 샘플 수로 가중해야 한다.** 단순 평균은 느린 워커의 적은 샘플을 과대 반영한다.
2. **안 되면 훈련에서 제외** — `T_step > 3 × median`이거나 `cc < 7.0`(bf16/fp16 텐서코어 없음, 예: GTX 970). 전처리·토크나이즈로 재배치.

발표 대사: *"970을 훈련에 억지로 넣으면 전체가 970 속도가 됩니다. 빼서 전처리를 시키면 970도 기여하고 나머지는 빨라집니다."*

## 추론 레버

| # | 레버 | 규칙 | 효과 |
|---|---|---|---|
| 1 | vLLM (continuous batching, PagedAttention) | 동시성 > 1이면 항상 | 동시성이 높을수록 격차가 커진다. **동시성 1이면 차이 거의 없음** |
| 2 | 양자화 (int8 / AWQ / GPTQ) | VRAM 압박 또는 처리량 미달 | ~2× 처리량, 메모리 절반. perplexity 확인 |
| 3 | `gpu_memory_utilization` 상향 | KV 캐시 부족으로 큐 적체 | 동시 시퀀스↑ |
| 4 | `max_model_len` 축소 | 실제 프롬프트가 짧을 때 | KV 캐시 여유↑ |
| 5 | 칩 라이트사이징 | 구세대에도 들어갈 때 | 크레딧↓, 속도 소폭↓ |

**레버 1의 조건이 데모 설계를 결정한다: 고동시성(예: 64)으로 측정할 것.** 동시성 1이면 vLLM의 장점이 안 나타난다.

**발표 참고:** DiLoCo는 추론과 무관하다 — 훈련 알고리즘이다. 여기서 분산 추론은 요청 단위 독립 라우팅이고, 각 요청을 GPU 하나가 통째로 처리한다. 그래서 추론 쪽에선 벤더가 섞여도 무관하다.

## before/after 측정

### 훈련 — 프로브 측정 + 투명한 외삽

5시간을 두 번 돌릴 수 없다. 짧게 재서 길게 환산하되, **환산을 화면에 보여준다.**

```
1. baseline 설정으로 40스텝
     → 앞 10 버림 (autotune, allocator warmup, compile)
     → 나머지 30스텝의 중앙값
2. optimized 설정으로 40스텝, 동일 seed·데이터 순서
3. tokens/sec = global_batch_tokens / t_step   ← 이걸로 비교
4. ETA = total_steps × t_step + (total_steps / H) × t_sync
5. 크레딧 = GPU-시간 × 단가, 양쪽 다
```

```
┌──────────────────────────────────────────────┐
│  최적화 전/후        (Ampere24GB 에이전트)     │
│  설정         기본            에이전트         │
│  ───────────────────────────────────────      │
│  dtype        fp32           bf16             │
│  batch        8 (×4 누적)    32 (×1 누적)     │
│  attention    eager          sdpa             │
│  H            —              190              │
│                                               │
│  step당 토큰       32,768   32,768   ✓        │
│  tokens/sec        14,200   19,800            │
│  step time(중앙값) 2.31s    1.66s             │
│                                               │
│  측정: 30스텝 (warmup 10 제외)                 │
│  환산: 8,000 × 1.66s + 42 sync × 2.0s         │
│        = 5h 08m → 3h 42m  (투영값)            │
│  크레딧:  38 → 28                             │
│  loss 곡선 겹침: ●                            │
└──────────────────────────────────────────────┘
```

### 반칙 목록

| 반칙 | 왜 걸리나 |
|---|---|
| 배치 키우고 grad_accum 안 줄임 | 일을 덜 한 것. tokens/sec이 거의 안 오른다 |
| baseline에 warmup 포함 | baseline을 인위적으로 느리게. 가장 흔한 자기기만 |
| 평균 사용 | 이상치 하나에 흔들린다. 중앙값 |
| 일부러 못난 baseline (fp32 + 배치 1) | "아무도 그렇게 안 씁니다"에 끝. **baseline = 초보자의 합리적 기본값** |
| seed나 데이터 순서 다름 | loss 비교가 무의미해진다 |
| 투영값을 실측처럼 말함 | 치명적. 화면과 말 양쪽에 **투영**이라고 |

### 추론 — 실측

```
고정 부하:  프롬프트 64개 동시 전송, 각 출력 정확히 128토큰
           (ignore_eos=True, max_tokens=128  ← 출력량을 못으로 박는다)

A) baseline:  HuggingFace generate 순차 루프
B) optimized: vLLM

지표: 전체 세트 wall clock → output tokens/sec = (64 × 128) / elapsed
```

`ignore_eos`가 핵심이다. 없으면 A와 B가 다른 양의 토큰을 생성해서 비교가 무너진다. **추론판 "같은 일" 불변식이다.**

```
동시 요청 64개 · 출력 각 128토큰 (고정)

기본 (순차)      ████████████████████  41.2s
vLLM (continuous) ███                    6.8s

output tokens/sec:  199 → 1,204
출력 토큰: 8,192 = 8,192 ✓
```

막대 두 개가 자라는 걸 보여주면 설명이 필요 없다. 양자화까지 켰다면 샘플 출력 두 개를 나란히 놓아 "답이 망가졌냐"를 선제적으로 막는다.

## 최적화를 데모에 넣기

에이전트 장면 세 개(최적화, 회수, 마이그레이션)를 5분에 다 넣으면 터진다. **최적화와 마이그레이션을 하나의 서사로 묶는다:**

```
에이전트가 두 방법으로 답하는 하나의 질문: "더 빠르고 싸게, 어떻게?"
        ├─ 설정을 바꿔서 (칩 에이전트) → 5h 08m → 3h 42m · 38 → 28 cr
        └─ 칩을 바꿔서   (라우터)       → 3h 42m → 2h 50m · 28 → 24 cr
```

> *"에이전트는 하나의 질문에 두 가지로 답합니다. 설정을 바꾸거나, 칩을 바꾸거나. **어느 쪽이 나은지 정하는 게 저희가 만든 부분입니다.**"*

- **최적화 결정은 훈련 시작 *전* 30초 장면으로.** 별도 비트를 만들지 않는다.
- **추론 라이브 before/after는 별도 데모로.** Q&A나 부스용. 막대 두 개짜리 화면 하나면 30초면 된다.

## 구현 체크리스트 (Ji)

- [ ] `ChipAgent` 기반 클래스 + 최소 `Ampere24GB`와 `DefaultAgent`
- [ ] `Router.agent_for()` — `chip_class` 디스패치 (계약 5, Jack의 SQLite와 같은 문자열)
- [ ] 칩 종류가 섞인 풀용 `reconcile()`
- [ ] 프로브 러너 (중앙값 step time, peak mem, util, sync time)
- [ ] 배치 탐색 + OOM 롤백
- [ ] global batch 불변식 강제 (`micro_batch × grad_accum × workers = 상수`)
- [ ] tokens/sec 표시 (스텝/초 절대 금지)
- [ ] H 함수 + clamp + 상한 경고
- [ ] 워커별 step time → 낙오자 판정 → 균형 또는 재배치
- [ ] outer average의 샘플 수 가중 (Ethan과 합의)
- [ ] ETA 공식 + **투영값** 라벨
- [ ] 크레딧 비용을 시간 옆에, 계산식과 함께
- [ ] `pick_chips()` — 가장 빠른 게 아니라 충분한 것 중 가장 싼
- [ ] baseline을 초보자 기본값으로 정의, 로그 보존
- [ ] 추론 부하 생성기: 동시 64, `ignore_eos`, `max_tokens` 고정
