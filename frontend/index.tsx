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

interface Scenario {
  id: string;
  name: string; // Senaryo adı (Backend'den gelen bilgi)
  request: {
    terminalId: string;
    traceId: string;
    amount: number;
    currency: string;
    panToken: string;
    timestamp: number;
    nonce: string;
  };
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
  fraudResult: {
    fraudScore: number;
    riskLevel: RiskLevel;
    decision: Decision;
    reasons: string[];
  };
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

// --- MOCK BACKEND DATA STREAM ---
// Burası senin backend scriptinden gelecek olan JSON listesini temsil eder.
// Frontend bu veriyi değiştirmez, sadece sırasıyla okur ve oynatır.

const MOCK_DATA_STREAM: Scenario[] = [
  {
    id: "tx-1",
    name: "Scenario 1: Standard Morning Coffee",
    request: { terminalId: "TERM-001", traceId: "trc-1001", amount: 45.0, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735660200, nonce: "n7721a" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 8, isNight: 0, distanceKm: 0.5, amtZscore: -0.2, cardAvgAmt: 50, timeSinceLastTx: 3600 },
    fraudResult: { fraudScore: 0.01, riskLevel: "MINIMAL", decision: "APPROVED", reasons: [] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-2",
    name: "Scenario 2: Lunch at City Center",
    request: { terminalId: "TERM-004", traceId: "trc-1002", amount: 150.0, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735670000, nonce: "n7722b" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 13, isNight: 0, distanceKm: 2.1, amtZscore: 0.1, cardAvgAmt: 50, timeSinceLastTx: 12000 },
    fraudResult: { fraudScore: 0.03, riskLevel: "MINIMAL", decision: "APPROVED", reasons: [] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-3",
    name: "Scenario 3: High Value Electronics (Risk)",
    request: { terminalId: "TERM-099", traceId: "trc-1003", amount: 25000.0, currency: "TRY", panToken: "tok_master_8812", timestamp: 1735675000, nonce: "n9912b" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 15, isNight: 0, distanceKm: 15.0, amtZscore: 5.5, cardAvgAmt: 500, timeSinceLastTx: 120 },
    fraudResult: { fraudScore: 0.85, riskLevel: "HIGH", decision: "PENDING", reasons: ["High Amount", "Velocity Check"] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-4",
    name: "Scenario 4: Security Replay Attack",
    request: { terminalId: "TERM-XXX", traceId: "trc-bad-actor", amount: 1.0, currency: "TRY", panToken: "tok_cloned_000", timestamp: 1735660999, nonce: "used_nonce_123" },
    securityCheck: { mtls: true, headerHmac: false, nonce: false, timestamp: true, bodySignature: true }, // BACKEND SAYS FAIL
    features: { hour: 3, isNight: 1, distanceKm: 999, amtZscore: 0, cardAvgAmt: 0, timeSinceLastTx: 0 },
    fraudResult: { fraudScore: 0, riskLevel: "CRITICAL", decision: "DECLINED", reasons: ["Security Violation"] },
    persisted: false, fallbackUsed: false,
  },
  {
    id: "tx-5",
    name: "Scenario 5: Night Gas Station",
    request: { terminalId: "TERM-055", traceId: "trc-1005", amount: 800.0, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735688000, nonce: "n7725e" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 23, isNight: 1, distanceKm: 45.0, amtZscore: 1.2, cardAvgAmt: 50, timeSinceLastTx: 300 },
    fraudResult: { fraudScore: 0.45, riskLevel: "MEDIUM", decision: "APPROVED", reasons: ["Night Transaction"] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-6",
    name: "Scenario 6: Timeout Fallback",
    request: { terminalId: "TERM-012", traceId: "trc-1006", amount: 60.0, currency: "TRY", panToken: "tok_troy_1111", timestamp: 1735690000, nonce: "n1111f" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 10, isNight: 0, distanceKm: 1.0, amtZscore: 0, cardAvgAmt: 60, timeSinceLastTx: 5000 },
    fraudResult: { fraudScore: 0.1, riskLevel: "LOW", decision: "APPROVED", reasons: ["System Timeout", "Fallback Applied"] },
    persisted: true, fallbackUsed: true, // BACKEND SAYS FALLBACK USED
  },
  {
    id: "tx-7",
    name: "Scenario 7: E-Com Subscription",
    request: { terminalId: "VIRT-001", traceId: "trc-1007", amount: 199.90, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735700000, nonce: "n7727g" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 9, isNight: 0, distanceKm: 0, amtZscore: 0.5, cardAvgAmt: 50, timeSinceLastTx: 86400 },
    fraudResult: { fraudScore: 0.05, riskLevel: "MINIMAL", decision: "APPROVED", reasons: [] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-8",
    name: "Scenario 8: Stolen Card Attempt",
    request: { terminalId: "TERM-666", traceId: "trc-1008", amount: 5000.0, currency: "TRY", panToken: "tok_black_9999", timestamp: 1735705000, nonce: "n6666h" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 4, isNight: 1, distanceKm: 500.0, amtZscore: 10.0, cardAvgAmt: 100, timeSinceLastTx: 60 },
    fraudResult: { fraudScore: 0.98, riskLevel: "CRITICAL", decision: "DECLINED", reasons: ["Impossible Travel", "Amount Spike"] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-9",
    name: "Scenario 9: Grocery Shopping",
    request: { terminalId: "TERM-002", traceId: "trc-1009", amount: 340.50, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735710000, nonce: "n7729i" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 18, isNight: 0, distanceKm: 1.0, amtZscore: 0.8, cardAvgAmt: 50, timeSinceLastTx: 10000 },
    fraudResult: { fraudScore: 0.12, riskLevel: "LOW", decision: "APPROVED", reasons: [] },
    persisted: true, fallbackUsed: false,
  },
  {
    id: "tx-10",
    name: "Scenario 10: Late Night Taxi",
    request: { terminalId: "POS-TAXI", traceId: "trc-1010", amount: 220.0, currency: "TRY", panToken: "tok_visa_4421", timestamp: 1735720000, nonce: "n7730j" },
    securityCheck: { mtls: true, headerHmac: true, nonce: true, timestamp: true, bodySignature: true },
    features: { hour: 2, isNight: 1, distanceKm: 5.0, amtZscore: 0.3, cardAvgAmt: 50, timeSinceLastTx: 500 },
    fraudResult: { fraudScore: 0.25, riskLevel: "LOW", decision: "APPROVED", reasons: [] },
    persisted: true, fallbackUsed: false,
  },
];

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
  const [speed, setSpeed] = useState(2); // Varsayılan hızı biraz artırdık akış için
  const [logs, setLogs] = useState<string[]>([]);
  
  // Görsel animasyon state'leri (Data üretmez, sadece animasyon zamanlamasını yönetir)
  const [secCheckIndex, setSecCheckIndex] = useState(-1);
  const [scoreProgress, setScoreProgress] = useState(0);

  // MOCK STREAM'den sıradaki veriyi okuyoruz. Frontend karar vermez, backend'in verdiğini okur.
  const scenario = MOCK_DATA_STREAM[currentScenarioIdx];
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

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

  // State Machine: Sadece backend'den gelen veri durumlarını (States) görselleştirir.
  const nextStep = () => {
    switch (state) {
      case "IDLE":
        addLog(`REQ [${scenario.request.traceId}] <-- RECEIVED from ${scenario.request.terminalId}`);
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
        addLog(`Backend Fraud Score: ${scenario.fraudResult.fraudScore}`);
        setState("DECIDING");
        break;
      case "DECIDING":
        addLog(`Final Decision: ${scenario.fraudResult.decision}`);
        setState("PERSISTING");
        break;
      case "PERSISTING":
        if (scenario.persisted) addLog(`DB: Transaction saved.`);
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
    if (!isPlaying) return;

    let timer: ReturnType<typeof setTimeout>;
    const currentSpeed = isAutoPlay ? Math.max(speed, 2) : speed; 
    const baseDelay = 1000 / currentSpeed;

    const runLoop = async () => {
      if (state === "IDLE") {
        addLog(`REQ [${scenario.request.traceId}] <-- RECEIVED`);
        setState("VALIDATING");
      } 
      
      else if (state === "VALIDATING") {
        // Güvenlik kontrollerini tek tek ekrana çiziyoruz
        if (secCheckIndex < 4) {
          timer = setTimeout(() => {
            setSecCheckIndex((prev) => prev + 1);
            
            // Backend verisindeki sonucu kontrol et
            const checks = [
                scenario.securityCheck.mtls,
                scenario.securityCheck.headerHmac,
                scenario.securityCheck.nonce,
                scenario.securityCheck.timestamp,
                scenario.securityCheck.bodySignature
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
        if (scoreProgress < scenario.fraudResult.fraudScore * 100) {
            timer = setTimeout(() => {
                setScoreProgress(prev => Math.min(prev + 10, scenario.fraudResult.fraudScore * 100));
            }, 50 / currentSpeed);
        } else {
            timer = setTimeout(() => {
                if (scenario.fallbackUsed) addLog("WARN: Backend responded with Fallback.");
                setState("DECIDING");
            }, baseDelay);
        }
      } 
      
      else if (state === "DECIDING") {
        timer = setTimeout(() => {
          addLog(`Result: ${scenario.fraudResult.decision}`);
          setState("PERSISTING");
        }, baseDelay);
      } 
      
      else if (state === "PERSISTING") {
        timer = setTimeout(() => {
          if (scenario.persisted) addLog("Transaction persisted to storage.");
          setState("FINISHED");
        }, baseDelay);
      }

      else if (state === "FINISHED" || state === "ERROR") {
        // AUTO PLAY LOGIC: Sıradaki backend verisine geç
        if (isAutoPlay) {
            timer = setTimeout(() => {
                // Listede bir sonrakine geç (Döngüsel)
                setCurrentScenarioIdx(prev => (prev + 1) % MOCK_DATA_STREAM.length);
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
  }, [state, isPlaying, secCheckIndex, scoreProgress, scenario, speed, isAutoPlay]);


  // Görsel Yardımcılar
  const getSecurityStatus = (index: number) => {
    if (state === "IDLE") return null;
    if (state === "ERROR") {
        // Hata durumunda sadece hata noktasına kadar göster
        const checks = [scenario.securityCheck.mtls, scenario.securityCheck.headerHmac, scenario.securityCheck.nonce, scenario.securityCheck.timestamp, scenario.securityCheck.bodySignature];
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
            <div className={`p-2 rounded-lg shadow-lg transition-colors ${isAutoPlay ? "bg-red-600 shadow-red-500/20" : "bg-blue-600 shadow-blue-500/20"}`}>
                {isAutoPlay ? <Activity className="text-white w-6 h-6 animate-pulse" /> : <ShieldCheck className="text-white w-6 h-6" />}
            </div>
            <div>
                <h1 className="text-xl font-bold tracking-tight text-white">Sentinel <span className={isAutoPlay ? "text-red-500" : "text-blue-500"}>POS</span> Dashboard</h1>
                <p className="text-xs text-slate-400">
                    {isAutoPlay ? "Live Stream from Backend" : "Single Request View"}
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
                    disabled={isPlaying}
                >
                    {MOCK_DATA_STREAM.map((s, i) => (
                        <option key={s.id} value={i}>{s.name.substring(0, 35)}</option>
                    ))}
                </select>
            </div>

            <div className="flex items-center gap-2">
                {/* AUTO PLAY TOGGLE */}
                <button 
                    onClick={() => {
                        const newAuto = !isAutoPlay;
                        setIsAutoPlay(newAuto);
                        if (newAuto) {
                            if (state === "FINISHED" || state === "ERROR" || state === "IDLE") {
                                reset(false);
                                setIsPlaying(true);
                            }
                        } else {
                            setIsPlaying(false);
                        }
                    }}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded transition font-semibold text-xs border ${
                        isAutoPlay 
                        ? "bg-red-900/30 text-red-400 border-red-800 animate-pulse" 
                        : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700"
                    }`}
                >
                   {isAutoPlay ? <Zap size={14} /> : <ZapOff size={14} />}
                   {isAutoPlay ? "LIVE FEED" : "MANUAL"}
                </button>

                <div className="w-px h-6 bg-slate-700 mx-1"></div>

                <button 
                    onClick={() => {
                        if(state === "FINISHED" || state === "ERROR") reset(false);
                        setIsPlaying(!isPlaying);
                    }}
                    className={`p-2 rounded hover:bg-slate-700 transition ${isPlaying ? "bg-amber-500/10 text-amber-500" : "bg-emerald-500/10 text-emerald-500"}`}
                >
                    {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                </button>
                
                <button 
                    onClick={nextStep}
                    disabled={isPlaying || state === "FINISHED" || isAutoPlay}
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
                    <StatusBadge state={state === "IDLE" ? "PENDING" : state === "FINISHED" ? "SENT" : "PROCESSING"} />
                </div>
                
                <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-800 p-2 rounded">
                            <div className="text-[10px] text-slate-500">Terminal ID</div>
                            <div className="font-mono text-sm text-blue-200 transition-all">{scenario.request.terminalId}</div>
                        </div>
                        <div className="bg-slate-800 p-2 rounded">
                            <div className="text-[10px] text-slate-500">Amount</div>
                            <div className="font-mono text-sm text-emerald-300 transition-all">{scenario.request.amount} {scenario.request.currency}</div>
                        </div>
                    </div>
                    <div className="bg-slate-800 p-3 rounded font-mono text-xs text-slate-400 overflow-x-auto h-28">
                        <span className="text-purple-400">{"{"}</span><br/>
                        &nbsp;&nbsp;"traceId": <span className="text-yellow-300">"{scenario.request.traceId}"</span>,<br/>
                        &nbsp;&nbsp;"pan": <span className="text-yellow-300">"{scenario.request.panToken.substring(0,10)}..."</span>,<br/>
                        &nbsp;&nbsp;"nonce": <span className="text-yellow-300">"{scenario.request.nonce}"</span><br/>
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
                        value={scenario.features.hour + ":00"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Distance" 
                        value={scenario.features.distanceKm + " km"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Last Tx" 
                        value={scenario.features.timeSinceLastTx + "s"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Is Night" 
                        value={scenario.features.isNight ? "YES" : "NO"} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Amt Z-Score" 
                        value={scenario.features.amtZscore} 
                        highlight={state === "EXTRACTING"}
                    />
                    <FeatureCard 
                        label="Avg Amt" 
                        value={scenario.features.cardAvgAmt} 
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
                    {scenario.fallbackUsed && (state === "SCORING" || state === "DECIDING" || state === "FINISHED") && (
                        <div className="flex items-center gap-1 bg-amber-900/30 text-amber-500 px-2 py-1 rounded text-xs border border-amber-800">
                            <AlertTriangle size={12} /> FALLBACK MODE
                        </div>
                    )}
                </div>

                <div className="flex items-end justify-between mb-2">
                    <span className="text-slate-400 text-xs">Risk Score</span>
                    <span className="text-4xl font-bold text-white">
                        {(state === "SCORING" || state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") 
                            ? (state === "SCORING" ? scoreProgress.toFixed(0) : (scenario.fraudResult.fraudScore * 100).toFixed(0))
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
                            scenario.fraudResult.riskLevel === "CRITICAL" || scenario.fraudResult.riskLevel === "HIGH" ? "bg-red-900/40 text-red-400 border-red-800" :
                            scenario.fraudResult.riskLevel === "MEDIUM" ? "bg-yellow-900/40 text-yellow-400 border-yellow-800" :
                            "bg-emerald-900/40 text-emerald-400 border-emerald-800"
                         }`}>
                            RISK LEVEL: {scenario.fraudResult.riskLevel}
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
                        scenario.fraudResult.decision === "APPROVED" ? "border-emerald-500 bg-emerald-900/20 text-emerald-500" :
                        scenario.fraudResult.decision === "DECLINED" ? "border-red-500 bg-red-900/20 text-red-500" :
                        "border-yellow-500 bg-yellow-900/20 text-yellow-500"
                    }`}>
                        <div className="text-center transform -rotate-12">
                            {scenario.fraudResult.decision === "APPROVED" && <CheckCircle2 size={48} className="mx-auto mb-1" />}
                            {scenario.fraudResult.decision === "DECLINED" && <ShieldAlert size={48} className="mx-auto mb-1" />}
                            {scenario.fraudResult.decision === "PENDING" && <Activity size={48} className="mx-auto mb-1" />}
                            <span className="block font-bold text-sm tracking-wider">{scenario.fraudResult.decision}</span>
                        </div>
                    </div>
                ) : (
                    <div className="w-32 h-32 rounded-full border-4 border-slate-800 border-dashed flex items-center justify-center">
                        <span className="text-slate-600 text-xs">WAITING</span>
                    </div>
                )}

                {scenario.fraudResult.reasons.length > 0 && (state === "DECIDING" || state === "PERSISTING" || state === "FINISHED") && (
                    <div className="mt-6 flex flex-wrap gap-2 justify-center">
                        {scenario.fraudResult.reasons.map(r => (
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