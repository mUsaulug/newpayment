import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  ShieldCheck,
  ShieldAlert,
  Activity,
  Server,
  Database,
  CreditCard,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Lock,
  Wifi,
  Zap,
  Cpu,
  RefreshCw,
  ZapOff
} from "lucide-react";

// --- Types ---

type RiskLevel = "MINIMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
type Decision = "APPROVED" | "PENDING" | "DECLINED";
type TxnType = "AUTH" | "CAPTURE" | "VOID" | "REVERSAL";

interface PaymentResponse {
  approved: boolean;
  message: string;
  responseCode: string;
  fraudScore: number;
  riskLevel: RiskLevel;
  fraudReasons: string[];
}

const mapFraudDecision = (response: PaymentResponse): Decision => {
  if (response.approved || response.responseCode === "00") {
    return "APPROVED";
  }

  const message = response.message?.toUpperCase() ?? "";
  if (message.includes("PENDING") || response.responseCode === "01") {
    return "PENDING";
  }

  return "DECLINED";
};

const mapFraudResult = (response: PaymentResponse) => ({
  fraudScore: response.fraudScore,
  riskLevel: response.riskLevel,
  decision: mapFraudDecision(response),
  reasons: response.fraudReasons ?? [],
});

interface Scenario {
  id: string;
  name: string; // Senaryo adı (Backend'den gelen bilgi)
  request: {
    terminalId: string;
    traceId: string;
    txnType: TxnType;
    amount: number;
    currency: string;
    panToken: string;
    idempotencyKey: string;
  };
  demoPlaceholders?: {
    nonce: string;
    signature: string;
  };
  isDemo?: boolean;
  securityCheck: {
    mtls: boolean;
    headerHmac: boolean;
    nonce: boolean;
    timestamp: boolean;
    bodySignature: boolean;
  };
  features: {
    hour: number;
    isNight: number;
    distanceKm: number;
    amtZscore: number;
    cardAvgAmt: number;
    timeSinceLastTx: number;
  };
  response: PaymentResponse;
  persisted: boolean;
  fallbackUsed: boolean;
}

type SimulationState =
  | "IDLE"
  | "VALIDATING"
  | "EXTRACTING"
  | "SCORING"
  | "DECIDING"
  | "PERSISTING"
  | "FINISHED"
  | "ERROR";

// --- Scenario loader (manual JSON or live stream) ---
const DEFAULT_TXN_TYPE: TxnType = "AUTH";
const LIVE_STREAM_ENDPOINT = "/pos-client/stream";
const POS_PAYMENT_ENDPOINT = "/api/pos/payments";

type DemoScenarioPayload = {
  scenarios?: DemoScenario[];
};

type DemoScenario = {
  id: string;
  class: string;
  request: {
    traceId: string;
    panToken: string;
    amount: number;
    merchantLat: number;
    merchantLong: number;
    category: string;
    timestamp: number;
  };
  expected: {
    decision: Decision;
    shortReason: string;
    output: {
      securityCheck: {
        mtls: string;
        signature: string;
        timestampSkew: string;
        idempotency: string;
      };
      fraudScore: number;
      riskLevel: RiskLevel;
      decision: Decision;
      reasons: string[];
    };
  };
};

const DEMO_HOME = { lat: 40.9912, long: 29.0228 };

const toRadians = (value: number) => (value * Math.PI) / 180;

const toBase64Url = (bytes: Uint8Array) => {
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};

const randomBytes = (length: number) => {
  const bytes = new Uint8Array(length);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return bytes;
};

const generateNonce = () => toBase64Url(randomBytes(16));

const generateSignaturePlaceholder = () => toBase64Url(randomBytes(32));

const distanceKm = (lat1: number, lon1: number, lat2: number, lon2: number) => {
  const earthRadiusKm = 6371;
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Number((earthRadiusKm * c).toFixed(2));
};

const mapDecisionToResponseCode = (decision: Decision) => {
  switch (decision) {
    case "APPROVED":
      return "00";
    case "PENDING":
      return "01";
    default:
      return "05";
  }
};

const mapDemoScenario = (scenario: DemoScenario, index: number): Scenario => {
  const timestampMs = scenario.request.timestamp * 1000;
  const eventDate = Number.isFinite(timestampMs) ? new Date(timestampMs) : new Date();
  const hour = eventDate.getHours();
  const isNight = hour < 6 || hour >= 22 ? 1 : 0;
  const security = scenario.expected?.output?.securityCheck;
  const signaturePass = security?.signature?.toUpperCase() === "PASS";
  const demoPlaceholders = {
    nonce: generateNonce(),
    signature: generateSignaturePlaceholder(),
  };

  return {
    id: scenario.id,
    name: `${index + 1}. ${scenario.class} • ${scenario.request.category}`,
    request: {
      terminalId: `POS-${scenario.request.category.toUpperCase()}`,
      traceId: scenario.request.traceId,
      txnType: DEFAULT_TXN_TYPE,
      amount: scenario.request.amount,
      currency: "TRY",
      panToken: scenario.request.panToken,
      idempotencyKey: `idem-${scenario.request.traceId}`,
    },
    demoPlaceholders,
    isDemo: true,
    securityCheck: {
      mtls: security?.mtls?.toUpperCase() === "PASS",
      headerHmac: signaturePass,
      nonce: security?.idempotency?.toUpperCase() === "PASS",
      timestamp: security?.timestampSkew?.toUpperCase() === "PASS",
      bodySignature: signaturePass,
    },
    features: {
      hour,
      isNight,
      distanceKm: distanceKm(
        DEMO_HOME.lat,
        DEMO_HOME.long,
        scenario.request.merchantLat,
        scenario.request.merchantLong
      ),
      amtZscore: Number((scenario.request.amount / 1000).toFixed(2)),
      cardAvgAmt: Number((scenario.request.amount * 0.3 + 50).toFixed(2)),
      timeSinceLastTx: 3600,
    },
    response: {
      approved: scenario.expected.output.decision === "APPROVED",
      message: scenario.expected.output.decision,
      responseCode: mapDecisionToResponseCode(scenario.expected.output.decision),
      fraudScore: scenario.expected.output.fraudScore,
      riskLevel: scenario.expected.output.riskLevel,
      fraudReasons: scenario.expected.output.reasons ?? [],
    },
    persisted: scenario.expected.output.decision !== "DECLINED",
    fallbackUsed: false,
  };
};

const EMPTY_SCENARIO: Scenario = {
  id: "empty",
  name: "No scenarios loaded",
  request: {
    terminalId: "--",
    traceId: "--",
    txnType: DEFAULT_TXN_TYPE,
    amount: 0,
    currency: "TRY",
    panToken: "--",
    idempotencyKey: "--",
  },
  securityCheck: {
    mtls: false,
    headerHmac: false,
    nonce: false,
    timestamp: false,
    bodySignature: false,
  },
  features: {
    hour: 0,
    isNight: 0,
    distanceKm: 0,
    amtZscore: 0,
    cardAvgAmt: 0,
    timeSinceLastTx: 0,
  },
  response: {
    approved: false,
    message: "NO_DATA",
    responseCode: "--",
    fraudScore: 0,
    riskLevel: "LOW",
    fraudReasons: [],
  },
  persisted: false,
  fallbackUsed: false,
};

const normalizeScenario = (payload: Partial<Scenario>): Scenario => {
  const request = payload.request ?? EMPTY_SCENARIO.request;
  const traceId = request.traceId ?? `trace-${Date.now()}`;
  return {
    id: payload.id ?? `live-${traceId}`,
    name: payload.name ?? "Live Scenario",
    request: {
      terminalId: request.terminalId ?? "POS-LIVE",
      traceId,
      txnType: request.txnType ?? DEFAULT_TXN_TYPE,
      amount: request.amount ?? 0,
      currency: request.currency ?? "TRY",
      panToken: request.panToken ?? "--",
      idempotencyKey: request.idempotencyKey ?? `idem-${traceId}`,
    },
    demoPlaceholders: payload.demoPlaceholders,
    isDemo: payload.isDemo ?? false,
    securityCheck: payload.securityCheck ?? EMPTY_SCENARIO.securityCheck,
    features: payload.features ?? EMPTY_SCENARIO.features,
    response: payload.response ?? EMPTY_SCENARIO.response,
    persisted: payload.persisted ?? false,
    fallbackUsed: payload.fallbackUsed ?? false,
  };
};

// --- Components ---

const StatusBadge = ({ state }: { state: string }) => {
  const getColors = () => {
    switch (state) {
      case "IDLE": return "bg-slate-700 text-slate-300";
      case "ERROR": return "bg-red-900/50 text-red-300 border-red-700";
      case "FINISHED": 
      case "SENT":
        return "bg-emerald-900/50 text-emerald-300 border-emerald-700";
      default: return "bg-blue-900/50 text-blue-300 border-blue-700 animate-pulse";
    }
  };

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-mono border ${getColors()}`}>
      {state}
    </span>
  );
};

const SecurityItem = ({ label, status, active }: { label: string; status: boolean | null; active: boolean }) => {
  let icon = <div className="w-4 h-4 rounded-full border border-slate-600" />;
  let textClass = "text-slate-500";

  if (active) {
    icon = <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />;
    textClass = "text-blue-400 font-semibold";
  } else if (status === true) {
    icon = <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
    textClass = "text-emerald-400";
  } else if (status === false) {
    icon = <XCircle className="w-4 h-4 text-red-500" />;
    textClass = "text-red-400 font-semibold";
  }

  return (
    <div className={`flex items-center justify-between py-2 px-3 rounded-md transition-colors ${active ? "bg-slate-800" : ""}`}>
      <span className={`text-sm ${textClass}`}>{label}</span>
      {icon}
    </div>
  );
};

const FeatureCard = ({ label, value, highlight }: { label: string; value: string | number; highlight: boolean }) => (
  <div className={`flex flex-col p-2 rounded border transition-all duration-300 ${highlight ? "bg-indigo-900/30 border-indigo-500 scale-105" : "bg-slate-800 border-slate-700"}`}>
    <span className="text-[10px] uppercase tracking-wider text-slate-400">{label}</span>
    <span className={`text-lg font-mono font-medium ${highlight ? "text-indigo-300" : "text-slate-200"}`}>
      {value}
    </span>
  </div>
);

// --- Main Application ---

const App = () => {
  const [currentScenarioIdx, setCurrentScenarioIdx] = useState(0);
  const [state, setState] = useState<SimulationState>("IDLE");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAutoPlay, setIsAutoPlay] = useState(false);
  const [feedMode, setFeedMode] = useState<"manual" | "live">("manual");
  const [speed, setSpeed] = useState(2); // Varsayılan hızı biraz artırdık akış için
  const [logs, setLogs] = useState<string[]>([]);
  const [manualScenarios, setManualScenarios] = useState<Scenario[]>([]);
  const [liveScenarios, setLiveScenarios] = useState<Scenario[]>([]);
  const [dataError, setDataError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Görsel animasyon state'leri (Data üretmez, sadece animasyon zamanlamasını yönetir)
  const [secCheckIndex, setSecCheckIndex] = useState(-1);
  const [scoreProgress, setScoreProgress] = useState(0);

  const isLiveMode = feedMode === "live";
  const activeScenarios = isLiveMode ? liveScenarios : manualScenarios;
  const scenario = activeScenarios[currentScenarioIdx];
  const hasScenario = Boolean(scenario);
  const resolvedScenario = scenario ?? EMPTY_SCENARIO;
  const fraudResult = mapFraudResult(resolvedScenario.response);
  const showDemoPlaceholder = Boolean(resolvedScenario.demoPlaceholders) && !isLiveMode;
  const emptyMessage = !isLoading && !dataError && activeScenarios.length === 0
    ? (isLiveMode
        ? "Live mode has no data yet. Waiting for backend stream..."
        : "No scenarios found in demo_scenarios.json.")
    : null;
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    let isActive = true;

    const loadManualScenarios = async () => {
      setIsLoading(true);
      setDataError(null);
      try {
        const response = await fetch("/demo_scenarios.json");
        if (!response.ok) {
          throw new Error(`Scenario load failed (${response.status})`);
        }
        const payload = (await response.json()) as DemoScenarioPayload;
        const scenarios = payload?.scenarios;
        if (!scenarios || scenarios.length === 0) {
          throw new Error("Scenario list is empty.");
        }
        const mapped = scenarios.map(mapDemoScenario);
        if (isActive) {
          setManualScenarios(mapped);
          setCurrentScenarioIdx(0);
        }
      } catch (error) {
        if (isActive) {
          setManualScenarios([]);
          setDataError(
            "Manual demo data could not be loaded. Verify demo_scenarios.json format."
          );
        }
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    };

    if (!isLiveMode) {
      loadManualScenarios();
    }

    return () => {
      isActive = false;
    };
  }, [isLiveMode]);

  useEffect(() => {
    if (!isLiveMode) return;

    setLiveScenarios([]);
    setDataError(null);
    setIsLoading(true);

    const source = new EventSource(LIVE_STREAM_ENDPOINT);
    const handleLiveEvent = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data) as Partial<Scenario>;
        setLiveScenarios((prev) => [...prev, normalizeScenario(payload)]);
        setIsLoading(false);
      } catch (error) {
        setDataError("Live stream data could not be parsed.");
        setIsLoading(false);
      }
    };
    source.onmessage = handleLiveEvent;
    source.addEventListener("scenario", handleLiveEvent);
    source.onerror = () => {
      setDataError("Live stream connection failed. Check backend stream/pos-client.");
      setIsLoading(false);
      source.close();
    };

    return () => {
      source.removeEventListener("scenario", handleLiveEvent);
      source.close();
    };
  }, [isLiveMode]);

  const addLog = (msg: string) => {
    const time = new Date().toISOString().split("T")[1].slice(0, 8);
    setLogs((prev) => {
      // Hafıza şişmesin diye son 50 logu tut
      const newLogs = [...prev, `[${time}] ${msg}`];
      return newLogs.slice(-50);
    });
  };

  const reset = (fullReset = true) => {
    setState("IDLE");
    setSecCheckIndex(-1);
    setScoreProgress(0);
    if (fullReset) {
      setLogs([]);
      setIsPlaying(false);
      setIsAutoPlay(false);
    }
  };

  const sendPaymentRequest = async () => {
    if (!hasScenario) return;

    try {
      const response = await fetch(POS_PAYMENT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(resolvedScenario.request),
      });

      if (!response.ok) {
        addLog(`POS CLIENT ERROR: ${response.status} ${response.statusText}`);
        return;
      }

      const payload = (await response.json()) as PaymentResponse;
      addLog(`POS CLIENT OK: ${payload.responseCode} ${payload.message}`);
    } catch (error) {
      addLog("POS CLIENT ERROR: request failed.");
    }
  };

  // State Machine: Sadece backend'den gelen veri durumlarını (States) görselleştirir.
  const nextStep = () => {
    if (!hasScenario) {
      setDataError("No scenario data available to play.");
      return;
    }
    switch (state) {
      case "IDLE":
        addLog(`REQ [${resolvedScenario.request.traceId}] <-- RECEIVED from ${resolvedScenario.request.terminalId}`);
        sendPaymentRequest();
        setState("VALIDATING");
        break;
      case "VALIDATING":
        // Görsel olarak validasyon bitti kabul edip ilerletiyoruz
        if (secCheckIndex < 4) setIsPlaying(true); 
        break;
      case "EXTRACTING":
        addLog("Data enriched with backend features.");
        setState("SCORING");
        break;
      case "SCORING":
        addLog(`Backend Fraud Score: ${fraudResult.fraudScore}`);
        setState("DECIDING");
        break;
      case "DECIDING":
        addLog(`Final Decision: ${fraudResult.decision}`);
        setState("PERSISTING");
        break;
      case "PERSISTING":
        if (resolvedScenario.persisted) addLog(`DB: Transaction saved.`);
        else addLog("DB: Persistence skipped by backend rule.");
        setState("FINISHED");
        break;
      case "FINISHED":
      case "ERROR":
        reset();
        break;
    }
  };

  // The Visualization Loop (Animasyon Döngüsü)
  useEffect(() => {
    if (!isPlaying || !hasScenario) return;

    let timer: ReturnType<typeof setTimeout>;
    const currentSpeed = isAutoPlay ? Math.max(speed, 2) : speed; 
    const baseDelay = 1000 / currentSpeed;

    const runLoop = async () => {
      if (state === "IDLE") {
        addLog(`REQ [${resolvedScenario.request.traceId}] <-- RECEIVED`);
        sendPaymentRequest();
        setState("VALIDATING");
      } 
      
      else if (state === "VALIDATING") {
        // Güvenlik kontrollerini tek tek ekrana çiziyoruz
        if (secCheckIndex < 4) {
          timer = setTimeout(() => {
            setSecCheckIndex((prev) => prev + 1);
            
            // Backend verisindeki sonucu kontrol et
            const checks = [
                resolvedScenario.securityCheck.mtls,
                resolvedScenario.securityCheck.headerHmac,
                resolvedScenario.securityCheck.nonce,
                resolvedScenario.securityCheck.timestamp,
                resolvedScenario.securityCheck.bodySignature
            ];
            
            // Eğer backend bu adımda false göndermişse görsel olarak durdur
            if (!checks[secCheckIndex + 1]) {
                const checkNames = ["mTLS", "HMAC", "Nonce", "Time", "Body"];
                addLog(`SECURITY FAIL: ${checkNames[secCheckIndex + 1]} rejected by WAF.`);
                setState("ERROR");
            }

          }, baseDelay * 0.4);
        } else {
          timer = setTimeout(() => {
            addLog("Security checks passed.");
            setState("EXTRACTING");
          }, baseDelay);
        }
      } 
      
      else if (state === "EXTRACTING") {
        // Feature'lar zaten veride var, sadece ekranda gösterme süresi tanıyoruz
        timer = setTimeout(() => {
          setState("SCORING");
        }, baseDelay);
      } 
      
      else if (state === "SCORING") {
        // Skoru animasyonla yükselt (Verideki fraudScore hedefine kadar)
        if (scoreProgress < fraudResult.fraudScore * 100) {
            timer = setTimeout(() => {
                setScoreProgress(prev => Math.min(prev + 10, fraudResult.fraudScore * 100));
            }, 50 / currentSpeed);
        } else {
            timer = setTimeout(() => {
                if (resolvedScenario.fallbackUsed) addLog("WARN: Backend responded with Fallback.");
                setState("DECIDING");
            }, baseDelay);
        }
      } 
      
      else if (state === "DECIDING") {
        timer = setTimeout(() => {
          addLog(`Result: ${fraudResult.decision}`);
          setState("PERSISTING");
        }, baseDelay);
      } 
      
      else if (state === "PERSISTING") {
        timer = setTimeout(() => {
          if (resolvedScenario.persisted) addLog("Transaction persisted to storage.");
          setState("FINISHED");
        }, baseDelay);
      }

      else if (state === "FINISHED" || state === "ERROR") {
        // AUTO PLAY LOGIC: Sıradaki backend verisine geç
        if (isAutoPlay) {
            timer = setTimeout(() => {
                if (activeScenarios.length === 0) {
                  setIsPlaying(false);
                  setDataError("No scenarios available for auto play.");
                  return;
                }
                // Listede bir sonrakine geç (Döngüsel)
                setCurrentScenarioIdx(prev => (prev + 1) % activeScenarios.length);
                // State'leri sıfırla ama play modunu koru
                setState("IDLE");
                setSecCheckIndex(-1);
                setScoreProgress(0);
            }, 1500); // 1.5 saniye bekle yeni istek için
        } else {
            setIsPlaying(false);
        }
      }
    };

    runLoop();
    return () => clearTimeout(timer);
  }, [state, isPlaying, secCheckIndex, scoreProgress, scenario, speed, isAutoPlay, activeScenarios.length]);


  // Görsel Yardımcılar
  const getSecurityStatus = (index: number) => {
    if (state === "IDLE") return null;
    if (state === "ERROR") {
        // Hata durumunda sadece hata noktasına kadar göster
        const checks = [resolvedScenario.securityCheck.mtls, resolvedScenario.securityCheck.headerHmac, resolvedScenario.securityCheck.nonce, resolvedScenario.securityCheck.timestamp, resolvedScenario.securityCheck.bodySignature];
        if (index <= secCheckIndex) return checks[index];
        return null;
    }
    // Normal akış
    if (index < secCheckIndex) return true;
    if (index === secCheckIndex) return true;
    return null;
  };

  const getSecurityActive = (index: number) => {
    if (state !== "VALIDATING") return false;
    return index === secCheckIndex + 1;
  };


  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 flex flex-col gap-6">
      
      {/* --- HEADER --- */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between bg-slate-900 p-4 rounded-xl border border-slate-800 shadow-lg">
        <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg shadow-lg transition-colors ${isLiveMode ? "bg-red-600 shadow-red-500/20" : "bg-blue-600 shadow-blue-500/20"}`}>
                {isLiveMode ? <Activity className="text-white w-6 h-6 animate-pulse" /> : <ShieldCheck className="text-white w-6 h-6" />}
            </div>
            <div>
                <h1 className="text-xl font-bold tracking-tight text-white">Sentinel <span className={isLiveMode ? "text-red-500" : "text-blue-500"}>POS</span> Dashboard</h1>
                <p className="text-xs text-slate-400">
                    {isLiveMode ? "Live Stream from Backend" : "Manual JSON Scenarios"}
                </p>
            </div>
        </div>

        <div className="flex items-center gap-4 mt-4 md:mt-0 bg-slate-800/50 p-2 rounded-lg border border-slate-700">
            {/* Scenario Selector */}
            <div className="flex items-center gap-2 pr-4 border-r border-slate-700">
                <span className="text-xs font-medium text-slate-400 uppercase">Input Feed</span>
                <select 
                    className="bg-slate-900 border border-slate-700 text-sm rounded px-2 py-1 outline-none focus:border-blue-500 disabled:opacity-50"
                    value={currentScenarioIdx}
                    onChange={(e) => {
                        reset(true);
                        setCurrentScenarioIdx(Number(e.target.value));
                    }}
                    disabled={isPlaying || isLiveMode || activeScenarios.length === 0}
                >
                    {activeScenarios.map((s, i) => (
                        <option key={s.id} value={i}>{s.name.substring(0, 35)}</option>
                    ))}
                </select>
                <span className="text-[10px] text-slate-500">
                    {isLiveMode ? `Live: ${LIVE_STREAM_ENDPOINT}` : "Manual: /demo_scenarios.json"}
                </span>
            </div>

            <div className="flex items-center gap-2">
                {/* AUTO PLAY TOGGLE */}
                <button 
                    onClick={() => {
                        const nextMode = isLiveMode ? "manual" : "live";
                        setFeedMode(nextMode);
                        if (nextMode === "live") {
                            reset(false);
                            setIsAutoPlay(true);
                            setIsPlaying(true);
                        } else {
                            setIsAutoPlay(false);
                            setIsPlaying(false);
                        }
                    }}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded transition font-semibold text-xs border ${
                        isLiveMode 
                        ? "bg-red-900/30 text-red-400 border-red-800 animate-pulse" 
                        : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700"
                    }`}
                >
                   {isLiveMode ? <Zap size={14} /> : <ZapOff size={14} />}
                   {isLiveMode ? "LIVE FEED" : "MANUAL"}
                </button>

                <div className="w-px h-6 bg-slate-700 mx-1"></div>

                <button 
                    onClick={() => {
                        if(state === "FINISHED" || state === "ERROR") reset(false);
                        setIsPlaying(!isPlaying);
                    }}
                    disabled={!hasScenario || isLoading || Boolean(dataError)}
                    className={`p-2 rounded hover:bg-slate-700 transition ${isPlaying ? "bg-amber-500/10 text-amber-500" : "bg-emerald-500/10 text-emerald-500"}`}
                >
                    {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                </button>
                
                <button 
                    onClick={nextStep}
                    disabled={isPlaying || state === "FINISHED" || isAutoPlay || !hasScenario || isLoading || Boolean(dataError)}
                    className="p-2 rounded hover:bg-slate-700 text-slate-300 disabled:opacity-30"
                >
                    <SkipForward size={18} />
                </button>

                <button 
                    onClick={() => reset(true)}
                    className="p-2 rounded hover:bg-slate-700 text-slate-300"
                >
                    <RotateCcw size={18} />
                </button>
            </div>
            
            <div className="flex gap-1 ml-2">
                {[1, 2, 5].map(x => (
                    <button 
                        key={x}
                        onClick={() => setSpeed(x)}
                        className={`text-xs w-8 h-8 rounded border ${speed === x ? "bg-blue-600 border-blue-500 text-white" : "border-slate-700 text-slate-400 hover:bg-slate-700"}`}
                    >
                        {x}x
                    </button>
                ))}
            </div>
        </div>
      </header>

      {(isLoading || dataError || emptyMessage) && (
        <div className={`rounded-lg border px-4 py-2 text-sm ${
          dataError ? "bg-red-900/30 border-red-700 text-red-200" : "bg-slate-900 border-slate-700 text-slate-300"
        }`}>
          {isLoading && <span>Loading scenarios...</span>}
          {!isLoading && dataError && <span>{dataError}</span>}
          {!isLoading && !dataError && emptyMessage && <span>{emptyMessage}</span>}
        </div>
      )}

      {/* --- MAIN GRID --- */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        
        {/* LEFT COLUMN (Inputs & Checks) */}
        <div className="lg:col-span-3 flex flex-col gap-6">
            {/* POS Request Panel */}
            <div className={`bg-slate-900 rounded-xl border p-4 shadow-xl transition-all duration-500 ${state === "IDLE" ? "border-blue-500/50 shadow-blue-900/20" : "border-slate-800"}`}>
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <CreditCard size={16} /> POS Client Request
                    </h3>
                    {showDemoPlaceholder && (
                      <span className="text-[10px] uppercase tracking-widest text-amber-300 bg-amber-900/30 border border-amber-700 px-2 py-1 rounded">
                        demo placeholder
                      </span>
                    )}
                    <StatusBadge state={state === "IDLE" ? "PENDING" : state === "FINISHED" ? "SENT" : "PROCESSING"} />
                </div>
                
                <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-800 p-2 rounded">
                            <div className="text-[10px] text-slate-500">Terminal ID</div>
                            <div className="font-mono text-sm text-blue-200 transition-all">{resolvedScenario.request.terminalId}</div>
                        </div>
                        <div className="bg-slate-800 p-2 rounded">
                            <div className="text-[10px] text-slate-500">Amount</div>
                            <div className="font-mono text-sm text-emerald-300 transition-all">{resolvedScenario.request.amount} {resolvedScenario.request.currency}</div>
                        </div>
                    </div>
                    <div className="bg-slate-800 p-3 rounded font-mono text-xs text-slate-400 overflow-x-auto h-36">
                        <span className="text-purple-400">{"{"}</span><br/>
                        &nbsp;&nbsp;"terminalId": <span className="text-yellow-300">"{resolvedScenario.request.terminalId}"</span>,<br/>
                        &nbsp;&nbsp;"traceId": <span className="text-yellow-300">"{resolvedScenario.request.traceId}"</span>,<br/>
                        &nbsp;&nbsp;"amount": <span className="text-yellow-300">{resolvedScenario.request.amount}</span>,<br/>
                        &nbsp;&nbsp;"currency": <span className="text-yellow-300">"{resolvedScenario.request.currency}"</span>,<br/>
                        &nbsp;&nbsp;"panToken": <span className="text-yellow-300">"{resolvedScenario.request.panToken.substring(0,10)}..."</span>,<br/>
                        &nbsp;&nbsp;"idempotencyKey": <span className="text-yellow-300">"{resolvedScenario.request.idempotencyKey}"</span>,<br/>
                        &nbsp;&nbsp;"txnType": <span className="text-yellow-300">"{resolvedScenario.request.txnType}"</span>
                        <span className="text-purple-400">{"}"}</span>
                    </div>
                </div>
            </div>

            {/* Security Validation Panel */}
            <div className={`bg-slate-900 rounded-xl border p-4 shadow-xl transition-all duration-500 ${state === "VALIDATING" ? "border-amber-500/50 shadow-amber-900/20 ring-1 ring-amber-500/20" : state === "ERROR" ? "border-red-500" : "border-slate-800"}`}>
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <Lock size={16} /> WAF / Security
                    </h3>
                    {state === "VALIDATING" && <Activity size={16} className="text-amber-500 animate-pulse" />}
                </div>
                
                <div className="space-y-1">
                    <SecurityItem label="mTLS Handshake" status={getSecurityStatus(0)} active={getSecurityActive(0)} />
                    <SecurityItem label="Header HMAC" status={getSecurityStatus(1)} active={getSecurityActive(1)} />
                    <SecurityItem label="Nonce Format" status={getSecurityStatus(2)} active={getSecurityActive(2)} />
                    <SecurityItem label="Timestamp Skew" status={getSecurityStatus(3)} active={getSecurityActive(3)} />
                    <SecurityItem label="Body Signature" status={getSecurityStatus(4)} active={getSecurityActive(4)} />
                </div>
            </div>
        </div>

        {/* MIDDLE COLUMN (Processing) */}
        <div className="lg:col-span-6 flex flex-col gap-6">
            
            {/* Feature Extraction */}
            <div className={`bg-slate-900 rounded-xl border p-4 shadow-xl relative overflow-hidden transition-all duration-500 ${state === "EXTRACTING" ? "border-purple-500/50 shadow-purple-900/20" : "border-slate-800"}`}>
                 {state === "EXTRACTING" && (
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent animate-translateX" />
                 )}
                
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <Cpu size={16} /> Backend Features
                    </h3>
                    <span className={`text-xs ${state === "EXTRACTING" ? "text-purple-400" : "text-slate-600"}`}>
                        {state === "EXTRACTING" ? "Receiving..." : state === "SCORING" || state === "DECIDING" || state === "PERSISTING" || state === "FINISHED" ? "Received" : "Waiting"}
                    </span>
                </div>

                <div className="grid grid-cols-3 gap-3">
                    <FeatureCard 
                        label="Hour" 
                        value={resolvedScenario.features.hour + ":00"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Distance" 
                        value={resolvedScenario.features.distanceKm + " km"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Last Tx" 
                        value={resolvedScenario.features.timeSinceLastTx + "s"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Is Night" 
                        value={resolvedScenario.features.isNight ? "YES" : "NO"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Amt Z-Score" 
                        value={resolvedScenario.features.amtZscore} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Avg Amt" 
                        value={resolvedScenario.features.cardAvgAmt} 
                        highlight={state === "EXTRACTING"}
                    />
                </div>
            </div>

            {/* Fraud Scoring Engine */}
            <div className={`bg-slate-900 rounded-xl border p-6 shadow-xl relative transition-all duration-500 ${state === "SCORING" ? "border-cyan-500/50 shadow-cyan-900/20" : "border-slate-800"}`}>
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <Zap size={16} /> Fraud AI Response
                    </h3>
                    {resolvedScenario.fallbackUsed && (state === "SCORING" || state === "DECIDING" || state === "FINISHED") && (
                        <div className="flex items-center gap-1 bg-amber-900/30 text-amber-500 px-2 py-1 rounded text-xs border border-amber-800">
                            <AlertTriangle size={12} /> FALLBACK MODE
                        </div>
                    )}
                </div>

                <div className="flex items-end justify-between mb-2">
                    <span className="text-slate-400 text-xs">Risk Score</span>
                    <span className="text-4xl font-bold text-white">
                        {(state === "SCORING" || state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") 
                            ? (state === "SCORING" ? scoreProgress.toFixed(0) : (fraudResult.fraudScore * 100).toFixed(0))
                            : "--"}
                        <span className="text-base font-normal text-slate-500">/100</span>
                    </span>
                </div>

                {/* Gauge Bar */}
                <div className="h-4 bg-slate-800 rounded-full overflow-hidden relative">
                    <div 
                        className={`h-full transition-all duration-300 ease-out ${
                            scoreProgress > 70 ? "bg-red-500" : scoreProgress > 30 ? "bg-yellow-500" : "bg-emerald-500"
                        }`}
                        style={{ width: `${(state === "SCORING" || state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") ? scoreProgress : 0}%` }}
                    />
                </div>
                
                <div className="flex justify-between mt-2 text-[10px] text-slate-600 font-mono uppercase">
                    <span>Safe</span>
                    <span>Suspicious</span>
                    <span>Fraud</span>
                </div>

                {/* Risk Level Badge */}
                {(state === "SCORING" || state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") && scoreProgress > 0 && (
                    <div className="mt-4 flex justify-center">
                         <span className={`px-4 py-1 rounded-full text-xs font-bold border ${
                            fraudResult.riskLevel === "CRITICAL" || fraudResult.riskLevel === "HIGH" ? "bg-red-900/40 text-red-400 border-red-800" :
                            fraudResult.riskLevel === "MEDIUM" ? "bg-yellow-900/40 text-yellow-400 border-yellow-800" :
                            "bg-emerald-900/40 text-emerald-400 border-emerald-800"
                         }`}>
                            RISK LEVEL: {fraudResult.riskLevel}
                         </span>
                    </div>
                )}
            </div>

        </div>

        {/* RIGHT COLUMN (Decision & DB) */}
        <div className="lg:col-span-3 flex flex-col gap-6">
            
            {/* Decision Panel */}
            <div className={`flex-1 bg-slate-900 rounded-xl border p-4 shadow-xl flex flex-col items-center justify-center transition-all duration-500 ${state === "DECIDING" || state === "PERSISTING" || state === "FINISHED" ? "border-slate-600 opacity-100" : "border-slate-800 opacity-50"}`}>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Final Decision</h3>
                
                {(state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") ? (
                    <div className={`w-32 h-32 rounded-full border-4 flex items-center justify-center animate-in zoom-in duration-300 ${
                        fraudResult.decision === "APPROVED" ? "border-emerald-500 bg-emerald-900/20 text-emerald-500" :
                        fraudResult.decision === "DECLINED" ? "border-red-500 bg-red-900/20 text-red-500" :
                        "border-yellow-500 bg-yellow-900/20 text-yellow-500"
                    }`}>
                        <div className="text-center transform -rotate-12">
                            {fraudResult.decision === "APPROVED" && <CheckCircle2 size={48} className="mx-auto mb-1" />}
                            {fraudResult.decision === "DECLINED" && <ShieldAlert size={48} className="mx-auto mb-1" />}
                            {fraudResult.decision === "PENDING" && <Activity size={48} className="mx-auto mb-1" />}
                            <span className="block font-bold text-sm tracking-wider">{fraudResult.decision}</span>
                        </div>
                    </div>
                ) : (
                    <div className="w-32 h-32 rounded-full border-4 border-slate-800 border-dashed flex items-center justify-center">
                        <span className="text-slate-600 text-xs">WAITING</span>
                    </div>
                )}

                {fraudResult.reasons.length > 0 && (state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") && (
                    <div className="mt-6 flex flex-wrap gap-2 justify-center">
                        {fraudResult.reasons.map(r => (
                            <span key={r} className="px-2 py-0.5 bg-slate-800 text-slate-400 text-[10px] rounded border border-slate-700">
                                {r}
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* DB Persistence Log */}
            <div className={`bg-slate-950 rounded-xl border p-0 overflow-hidden shadow-xl flex flex-col h-48 transition-colors ${state === "PERSISTING" ? "border-blue-500" : "border-slate-800"}`}>
                <div className="bg-slate-900 px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                     <span className="text-xs font-mono text-slate-400 flex items-center gap-2"><Database size={12}/> backend_logs</span>
                     {state === "PERSISTING" && <div className="w-2 h-2 bg-green-500 rounded-full animate-ping" />}
                </div>
                <div className="p-3 font-mono text-[10px] text-slate-300 overflow-y-auto flex-1 space-y-1">
                    {logs.length === 0 && <span className="text-slate-600 italic">// Waiting for backend feed...</span>}
                    {logs.map((log, i) => (
                        <div key={i} className="border-l-2 border-slate-700 pl-2 opacity-90 animate-in slide-in-from-left-2 duration-200">
                            {log}
                        </div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>

        </div>
      </main>
    </div>
  );
};

const container = document.getElementById("root");
const root = createRoot(container!);
root.render(<App />);
