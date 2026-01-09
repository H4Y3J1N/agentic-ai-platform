# 🚀 무중단 배포 전략 - 완전 구현

## ✅ 구현된 무중단 배포 기능

### 1. **Health Check Probes** (K8s)
```yaml
# deployment.yaml
livenessProbe:   # Pod 재시작 기준
  httpGet:
    path: /health
  failureThreshold: 3

readinessProbe:  # 트래픽 라우팅 기준
  httpGet:
    path: /readiness
  failureThreshold: 2

startupProbe:    # 초기화 완료 기준
  httpGet:
    path: /startup
  failureThreshold: 30
```

**구현 파일**: `services/ecommerce/api/routes/health.py`

### 2. **Graceful Shutdown** (SIGTERM)
```python
# main.py
def handle_sigterm(signum, frame):
    health.set_shutdown()  # 새 요청 차단
    await graceful_shutdown()  # 30초 Connection Draining
```

**프로세스**:
1. K8s가 SIGTERM 전송
2. `/readiness` → 503 반환 (트래픽 차단)
3. 30초 대기 (기존 요청 처리)
4. Pod 종료

### 3. **RollingUpdate 전략**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1          # 최대 1개 추가 Pod
    maxUnavailable: 0    # 항상 최소 replicas 유지
```

**동작**:
- Old Pod 1개 종료 전에 New Pod 1개 Ready 대기
- 트래픽 무중단 전환
- 전체 배포 완료까지 최소 3개 Pod 유지

### 4. **PodDisruptionBudget (PDB)**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
spec:
  minAvailable: 2  # 최소 2개 Pod 항상 유지
```

**효과**:
- 노드 유지보수 시에도 최소 Pod 수 보장
- Voluntary Disruption 방지

### 5. **PreStop Hook**
```yaml
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 30"]
```

**효과**:
- SIGTERM 전송 전 30초 대기
- Load Balancer가 트래픽 라우팅 중단할 시간 확보

### 6. **Session Affinity** (WebSocket)
```yaml
# service.yaml
sessionAffinity: ClientIP
sessionAffinityConfig:
  clientIP:
    timeoutSeconds: 3600
```

**효과**:
- WebSocket 연결이 같은 Pod로 유지
- 무중단 배포 시 새 연결만 새 Pod로

---

## 🎯 Canary 배포 (Argo Rollouts)

### Canary 단계별 트래픽 전환
```yaml
strategy:
  canary:
    steps:
    - setWeight: 10   # 10% 트래픽 → 5분 대기
    - setWeight: 30   # 30% 트래픽 → 5분 대기
    - setWeight: 50   # 50% 트래픽 → 5분 대기
    - setWeight: 80   # 80% 트래픽 → 5분 대기
    # 100% 자동 전환
```

### 자동 롤백 (Analysis)
```yaml
analysis:
  metrics:
  - name: success-rate
    successCondition: result >= 0.95  # 95% 이상
    failureLimit: 3  # 3번 실패 시 자동 롤백
```

**파일**: `infrastructure/k8s/services/ecommerce/rollout.yaml`

---

## 🗄️ DB Migration 무중단 전략

### Backward Compatible Migration
```python
# 1. 컬럼 추가
ALTER TABLE users ADD COLUMN new_field VARCHAR(255) NULL;  # NULL 허용

# 2. 애플리케이션 배포 (new_field 사용)

# 3. 기본값 채우기
UPDATE users SET new_field = 'default' WHERE new_field IS NULL;

# 4. NOT NULL 제약 추가 (선택)
ALTER TABLE users ALTER COLUMN new_field SET NOT NULL;
```

### 컬럼 삭제 전략
```
Phase 1: 애플리케이션에서 컬럼 사용 중단 (배포)
Phase 2: 컬럼 삭제 (배포)
```

**파일**: `scripts/migrations/backward_compatible_migration.py`

---

## 📊 배포 프로세스 (전체)

```
┌─────────────────────────────────────────────────────┐
│ 1. Pre-deployment Check                             │
│    - K8s 클러스터 상태 확인                           │
│    - 현재 Deployment 존재 확인                        │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 2. DB Migration (Backward Compatible)               │
│    - Backward compatibility 검증                     │
│    - Migration 실행                                  │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 3. Build & Push Image                               │
│    - Docker 이미지 빌드                               │
│    - Registry에 Push                                 │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 4. Rolling Update                                    │
│    Old Pod: 3개 → 2개 → 1개 → 0개                   │
│    New Pod: 0개 → 1개 → 2개 → 3개                   │
│                                                      │
│    각 단계마다:                                       │
│    - startupProbe 성공 대기                          │
│    - readinessProbe 성공 대기                        │
│    - 트래픽 전환                                      │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 5. Health Check (5분)                               │
│    - Error Rate 모니터링                             │
│    - 5% 이상 에러 시 자동 롤백                        │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 6. Cleanup                                          │
│    - Old ReplicaSet 삭제                            │
│    - 배포 완료 로그                                  │
└─────────────────────────────────────────────────────┘
```

---

## 🚨 장애 시 롤백 시나리오

### 자동 롤백 조건
1. **startupProbe 실패** → 30번 실패 (150초) → Pod 재시작
2. **readinessProbe 실패** → 트래픽 차단 → 새 Pod 생성
3. **livenessProbe 실패** → 3번 실패 → Pod 재시작
4. **Error Rate > 5%** → 자동 롤백 (Canary 배포)

### 수동 롤백
```bash
kubectl rollout undo deployment/ecommerce-ai -n agentic-ai
kubectl rollout undo deployment/ecommerce-ai -n agentic-ai --to-revision=2
```

---

## 📈 무중단 배포 검증

### 1. 배포 중 트래픽 테스트
```bash
# 배포 시작
kubectl set image deployment/ecommerce-ai api=ecommerce-ai:v2

# 동시에 부하 테스트
while true; do
  curl http://ecommerce-ai/health
  sleep 0.1
done
```

**기대 결과**: 200 OK만 반환 (502, 503 없음)

### 2. WebSocket 연결 유지
```bash
# WebSocket 연결 후 배포
wscat -c ws://ecommerce-ai/ws/chat/session123

# 배포 중에도 연결 유지됨 (Session Affinity)
```

### 3. 에러율 모니터링
```promql
# Prometheus Query
sum(rate(http_requests_total{service="ecommerce-ai",status=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="ecommerce-ai"}[5m]))
```

---

## ✅ 무중단 배포 체크리스트

### 애플리케이션 레벨
- [x] Health Check 엔드포인트 (/health, /readiness, /startup)
- [x] Graceful Shutdown (SIGTERM 핸들러)
- [x] Connection Draining (30초)
- [x] 의존성 체크 (DB, Redis, Milvus)

### K8s 레벨
- [x] RollingUpdate 전략 (maxSurge, maxUnavailable)
- [x] Health Probes (liveness, readiness, startup)
- [x] PodDisruptionBudget (minAvailable: 2)
- [x] PreStop Hook (30초 대기)
- [x] Resource Limits (OOMKilled 방지)
- [x] HPA (Auto Scaling)

### 배포 전략
- [x] Canary 배포 (Argo Rollouts)
- [x] 자동 롤백 (Analysis Template)
- [x] Traffic Splitting (Istio)

### DB Migration
- [x] Backward Compatible 전략
- [x] Phase별 마이그레이션
- [x] 롤백 가능 설계

---

## 🎓 무중단 배포 Best Practices

1. **항상 Backward Compatible 유지**
   - 새 버전이 구 DB 스키마와 호환
   - 구 버전이 새 DB 스키마와 호환

2. **Health Check 신뢰성**
   - 의존성 서비스까지 체크
   - 빠른 응답 (<3초)

3. **적절한 Timeout 설정**
   - terminationGracePeriodSeconds: 60s
   - readinessProbe timeout: 3s
   - PreStop hook: 30s

4. **모니터링 필수**
   - Error Rate
   - Latency (P50, P95, P99)
   - Active Connections

5. **단계적 배포**
   - Dev → Staging → Production
   - Canary 10% → 30% → 50% → 100%
