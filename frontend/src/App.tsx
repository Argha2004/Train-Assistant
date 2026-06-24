import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Upload,
  BarChart2,
  Cpu,
  Settings,
  Plus,
  Trash2,
  Download,
  RefreshCw,
  CheckCircle2,
  FileText,
  Send,
  Loader,
  TrendingUp,
  BookOpen,
  X,
  Activity,
  Thermometer,
  Zap,
  GitBranch,
  Menu,
  Volume2,
  VolumeX
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Cell,
  Pie
} from 'recharts';
import ReactMarkdown from 'react-markdown';

// Types
interface Message {
  role: 'user' | 'model' | 'tool';
  content?: string;
  tool_calls?: { name: string; args: any }[];
  tool_responses?: { name: string; response: any }[];
}

interface Chat {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

interface UploadItem {
  id: string;
  filename: string;
  size_bytes: number;
  columns: string[];
  uploaded_at: string;
}

interface ReportItem {
  id: string;
  name: string;
  filename: string;
  path: string;
  created_at: string;
}

interface SettingsState {
  gemini_model: string;
  theme: string;
  system_prompt: string;
  upload_dir: string;
  report_dir: string;
  auto_save: boolean;
  streaming_toggle: boolean;
  gemini_api_key?: string;
}

interface GPUInfo {
  gpu_id: number;
  name: string;
  vram_total_gb: number;
  vram_used_gb: number;
  vram_free_gb: number;
  vram_utilization_pct: number;
  gpu_utilization_pct: number;
  temperature_c: number;
  estimated_batch_size_multiplier: number;
}

interface DashboardStats {
  num_uploads: number;
  num_reports: number;
  average_auc: number;
  latest_experiments: {
    name: string;
    filename: string;
    epochs: number;
    best_auc: number | null;
    best_val_loss: number | null;
    uploaded_at: string;
  }[];
  recent_recommendations: {
    metric: string;
    recommendation: string;
    severity: string;
  }[];
}

interface ToolStatus {
  name: string;
  status: 'running' | 'completed' | 'failed';
  arguments?: any;
  result?: any;
  error?: string;
}

class SoundEffects {
  private static ctx: AudioContext | null = null;
  private static enabled = true;

  static setEnabled(val: boolean) {
    this.enabled = val;
  }

  private static initCtx() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  static playClick() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(1200, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(100, ctx.currentTime + 0.05);

      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.05);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  }

  static playOpen() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const now = ctx.currentTime;

      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(220, now);
      osc1.frequency.exponentialRampToValueAtTime(880, now + 0.15);

      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(330, now);
      osc2.frequency.exponentialRampToValueAtTime(1320, now + 0.15);

      gain.gain.setValueAtTime(0.1, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.15);
      osc2.stop(now + 0.15);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  }

  static playLoading() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(400, now);
      osc.frequency.linearRampToValueAtTime(800, now + 0.15);

      gain.gain.setValueAtTime(0.08, now);
      gain.gain.linearRampToValueAtTime(0.001, now + 0.15);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.15);
    } catch (e) {
      console.warn("Audio error:", e);
    }
  }

  static playSuccess() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const now = ctx.currentTime;

      const freqs = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6 (Bright C Major Chord)
      freqs.forEach((f, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(f, now + idx * 0.05);

        gain.gain.setValueAtTime(0.08, now + idx * 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.05 + 0.35);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now + idx * 0.05);
        osc.stop(now + idx * 0.05 + 0.35);
      });
    } catch (e) {
      console.warn("Audio error:", e);
    }
  }

  static playAiChirp() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(1000 + Math.random() * 500, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.02);

      gain.gain.setValueAtTime(0.02, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.02);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.02);
    } catch (e) { }
  }

  static playLoaderTick() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(180, now + 0.12);

      gain.gain.setValueAtTime(0.015, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.12);
    } catch (e) { }
  }

  static playError() {
    if (!this.enabled) return;
    try {
      const ctx = this.initCtx();
      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(180, now);
      osc1.frequency.linearRampToValueAtTime(120, now + 0.25);

      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(185, now);
      osc2.frequency.linearRampToValueAtTime(125, now + 0.25);

      gain.gain.setValueAtTime(0.12, now);
      gain.gain.linearRampToValueAtTime(0.001, now + 0.25);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.25);
      osc2.stop(now + 0.25);
    } catch (e) { }
  }
}

const HELLOS = [
  { text: "Hello", lang: "English" },
  { text: "Hola", lang: "Spanish" },
  { text: "Bonjour", lang: "French" },
  { text: "Hallo", lang: "German" },
  { text: "Ciao", lang: "Italian" },
  { text: "नमस्ते", lang: "Hindi" },
  { text: "নমস্কার", lang: "Bengali" },
  { text: "こんにちは", lang: "Japanese" },
  { text: "你好", lang: "Chinese" },
  { text: "Привет", lang: "Russian" },
  { text: "مرحبا", lang: "Arabic" },
  { text: "Olá", lang: "Portuguese" },
  { text: "안녕하세요", lang: "Korean" }
];

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [showLoader, setShowLoader] = useState(true);
  const [loaderIndex, setLoaderIndex] = useState(0);
  const [loaderFadingOut, setLoaderFadingOut] = useState(false);
  const [activeView, setActiveView] = useState<'dashboard' | 'chat' | 'comparison' | 'dataset' | 'gpu' | 'settings'>('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(() => {
    const saved = localStorage.getItem('sound_enabled');
    return saved !== null ? saved === 'true' : true;
  });

  useEffect(() => {
    SoundEffects.setEnabled(soundEnabled);
  }, [soundEnabled]);
  const [conversations, setConversations] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [settings, setSettings] = useState<SettingsState>({
    gemini_model: 'gemini-2.5-flash',
    theme: 'dark',
    system_prompt: '',
    upload_dir: './uploads',
    report_dir: './reports',
    auto_save: true,
    streaming_toggle: true,
    gemini_api_key: ''
  });
  const [dashboardStats, setDashboardStats] = useState<DashboardStats>({
    num_uploads: 0,
    num_reports: 0,
    average_auc: 0,
    latest_experiments: [],
    recent_recommendations: []
  });
  const [gpus, setGpus] = useState<GPUInfo[]>([]);
  const [gpuLoading, setGpuLoading] = useState(false);
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [previewReportUrl, setPreviewReportUrl] = useState<string | null>(null);
  const [previewReportName, setPreviewReportName] = useState<string>('');
  const [previewReportContent, setPreviewReportContent] = useState<string>('');
  const [selectedDatasetFile, setSelectedDatasetFile] = useState<string>('');
  const [datasetAnalysisResult, setDatasetAnalysisResult] = useState<any | null>(null);
  const [datasetAnalysisLoading, setDatasetAnalysisLoading] = useState<boolean>(false);


  // Chat stream states
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentTools, setCurrentTools] = useState<ToolStatus[]>([]);
  const [streamedText, setStreamedText] = useState('');

  // Ref for auto scroll
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Loading screen sequence
  useEffect(() => {
    SoundEffects.playLoaderTick();
    const interval = setInterval(() => {
      setLoaderIndex((prev) => {
        if (prev < HELLOS.length - 1) {
          SoundEffects.playLoaderTick();
          return prev + 1;
        } else {
          clearInterval(interval);
          SoundEffects.playSuccess();
          setLoaderFadingOut(true);
          setTimeout(() => {
            setShowLoader(false);
          }, 500);
          return prev;
        }
      });
    }, 450);

    return () => clearInterval(interval);
  }, []);

  // Load basic data
  useEffect(() => {
    fetchConversations();
    fetchUploads();
    fetchReports();
    fetchSettings();
    fetchDashboardStats();
    fetchGPUInfo();
  }, []);

  // Poll GPU info if view is active
  useEffect(() => {
    let interval: any;
    if (activeView === 'gpu') {
      fetchGPUInfo();
      interval = setInterval(fetchGPUInfo, 3000);
    }
    return () => clearInterval(interval);
  }, [activeView]);

  // Scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamedText, currentTools]);

  const fetchConversations = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchUploads = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/uploads`);
      if (res.ok) {
        const data = await res.json();
        setUploads(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchReports = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reports`);
      if (res.ok) {
        const data = await res.json();
        setReports(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`);
      if (res.ok) {
        const data = await res.json();
        setSettings(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchDashboardStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/dashboard`);
      if (res.ok) {
        const data = await res.json();
        setDashboardStats(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchGPUInfo = async () => {
    setGpuLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/gpu`);
      if (res.ok) {
        const data = await res.json();
        if (data.gpus) setGpus(data.gpus);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setGpuLoading(false);
    }
  };

  const startNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setStreamedText('');
    setCurrentTools([]);
    setActiveView('chat');
    setIsSidebarOpen(false);
    SoundEffects.playClick();
  };

  const selectChat = async (cid: string) => {
    setActiveChatId(cid);
    setActiveView('chat');
    setIsSidebarOpen(false);
    SoundEffects.playClick();
    try {
      const res = await fetch(`${API_BASE}/api/history/${cid}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
        setStreamedText('');
        setCurrentTools([]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const deleteChat = async (e: React.MouseEvent, cid: string) => {
    e.stopPropagation();
    SoundEffects.playClick();
    try {
      const res = await fetch(`${API_BASE}/api/history/${cid}`, { method: 'DELETE' });
      if (res.ok) {
        if (activeChatId === cid) startNewChat();
        fetchConversations();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || chatInput;
    if (!text.trim() || isGenerating) return;

    SoundEffects.playClick();
    setChatInput('');
    setIsGenerating(true);
    setStreamedText('');
    setCurrentTools([]);

    const updatedMsgs: Message[] = [...messages, { role: 'user', content: text }];
    setMessages(updatedMsgs);

    const tempChatId = activeChatId || crypto.randomUUID();
    if (!activeChatId) {
      setActiveChatId(tempChatId);
    }

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: tempChatId,
          message: text,
          model: settings.gemini_model,
          system_prompt: settings.system_prompt || undefined
        })
      });

      if (!response.ok) {
        let errMsg = `Server returned status ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errMsg = errData.detail;
          }
        } catch (_) {
          try {
            const errText = await response.text();
            if (errText) errMsg = errText;
          } catch (_) { }
        }
        throw new Error(errMsg);
      }

      if (!response.body) throw new Error("No readable stream in response");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6).trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'tool_start') {
                SoundEffects.playLoading();
                setCurrentTools(prev => [
                  ...prev.filter(t => t.name !== data.tool_name),
                  { name: data.tool_name, status: 'running', arguments: data.arguments }
                ]);
              } else if (data.type === 'tool_end') {
                SoundEffects.playSuccess();
                setCurrentTools(prev => [
                  ...prev.filter(t => t.name !== data.tool_name),
                  { name: data.tool_name, status: 'completed', result: data.result }
                ]);
                fetchReports();
                fetchUploads();
                fetchDashboardStats();
              } else if (data.type === 'text') {
                setStreamedText(prev => prev + data.content);
                SoundEffects.playAiChirp();
              } else if (data.type === 'error') {
                SoundEffects.playError();
                setStreamedText(prev => prev + `\n\n⚠️ ${data.message}\n`);
              } else if (data.type === 'done') {
                SoundEffects.playSuccess();
                const res = await fetch(`${API_BASE}/api/history/${tempChatId}`);
                if (res.ok) {
                  const chatData = await res.json();
                  setMessages(chatData.messages || []);
                }
                setStreamedText('');
                setCurrentTools([]);
                fetchConversations();
              }
            } catch (e) {
              console.error("Error parsing JSON block", e);
            }
          }
        }
      }
      // If stream ended without a 'done' event (e.g. server error), reload conversation
      if (streamedText || currentTools.length > 0) {
        try {
          const res = await fetch(`${API_BASE}/api/history/${tempChatId}`);
          if (res.ok) {
            const chatData = await res.json();
            setMessages(chatData.messages || []);
          }
        } catch (_) { }
        setStreamedText('');
        setCurrentTools([]);
        fetchConversations();
      }
    } catch (err: any) {
      console.error(err);
      SoundEffects.playError();
      setMessages(prev => [...prev, { role: 'model', content: `⚠️ **Connection Error:** ${err.message}. Make sure the backend server is running.` }]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    SoundEffects.playLoading();
    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        console.log("Uploaded successfully:", data);
        SoundEffects.playSuccess();
        fetchUploads();
        fetchDashboardStats();
        setActiveView('chat');
        handleSendMessage(`I just uploaded the training log file: ${file.name}. Can you inspect the contents, read the training logs, and summarize the best epoch?`);
      } else {
        const err = await res.json();
        SoundEffects.playError();
        alert(`Upload failed: ${err.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error(err);
      SoundEffects.playError();
      alert("Error uploading file.");
    }
  };

  const toggleRunSelection = (filename: string) => {
    SoundEffects.playClick();
    setSelectedRuns(prev =>
      prev.includes(filename) ? prev.filter(f => f !== filename) : [...prev, filename]
    );
  };

  const runComparison = async () => {
    if (selectedRuns.length === 0) return;
    SoundEffects.playLoading();
    setComparisonLoading(true);
    try {
      const dataPoints = [];
      for (let i = 0; i <= 20; i++) {
        dataPoints.push({
          epoch: i,
          run_a_loss: Number((0.85 * Math.pow(0.85, i) + 0.05 * (i > 12 ? (i - 12) * 0.08 : 0)).toFixed(4)),
          run_b_loss: Number((0.85 * Math.pow(0.82, i) + 0.03).toFixed(4))
        });
      }
      setComparisonData(dataPoints);

      setActiveView('chat');
      handleSendMessage(`Compare the following training runs: ${selectedRuns.join(', ')}. Generate a comparison table listing Best AUC, Best Epoch, and tell me which one is the winner!`);
    } catch (e) {
      console.error(e);
    } finally {
      setComparisonLoading(false);
    }
  };

  const previewReport = async (report: ReportItem) => {
    SoundEffects.playOpen();
    setPreviewReportName(report.name);
    setPreviewReportUrl(`${API_BASE}/api/reports-files/${report.filename}`);
    try {
      const res = await fetch(`${API_BASE}/api/reports-files/${report.filename}`);
      if (res.ok) {
        const text = await res.text();
        setPreviewReportContent(text);
      }
    } catch (e) {
      setPreviewReportContent("Error loading report content.");
    }
  };

  const deleteReport = async (rid: string) => {
    SoundEffects.playClick();
    try {
      const res = await fetch(`${API_BASE}/api/reports/${rid}`, { method: 'DELETE' });
      if (res.ok) {
        setPreviewReportUrl(null);
        fetchReports();
        fetchDashboardStats();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const deleteUpload = async (e: React.MouseEvent, uid: string) => {
    e.stopPropagation();
    SoundEffects.playClick();
    try {
      const res = await fetch(`${API_BASE}/api/uploads/${uid}`, { method: 'DELETE' });
      if (res.ok) {
        fetchUploads();
        fetchDashboardStats();
        // Clear selected file if deleted
        if (selectedDatasetFile && uploads.find(u => u.id === uid)?.filename === selectedDatasetFile) {
          setSelectedDatasetFile('');
          setDatasetAnalysisResult(null);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleAnalyzeDataset = async (filename: string) => {
    if (!filename) return;
    SoundEffects.playLoading();
    setDatasetAnalysisLoading(true);
    setDatasetAnalysisResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/analyze-dataset/${filename}`);
      if (res.ok) {
        const data = await res.json();
        SoundEffects.playSuccess();
        setDatasetAnalysisResult(data);
      } else {
        SoundEffects.playError();
        alert("Failed to analyze dataset.");
      }
    } catch (e) {
      console.error(e);
      SoundEffects.playError();
      alert("Error analyzing dataset.");
    } finally {
      setDatasetAnalysisLoading(false);
    }
  };


  const saveSettings = async (updated: Partial<SettingsState>) => {
    const newSettings = { ...settings, ...updated };
    setSettings(newSettings);
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings)
      });
    } catch (e) {
      console.error(e);
    }
  };

  const navItems = [
    { id: 'dashboard' as const, label: 'Overview', icon: BarChart2 },
    { id: 'chat' as const, label: 'Chat', icon: MessageSquare },
    { id: 'gpu' as const, label: 'GPU', icon: Cpu },
    { id: 'comparison' as const, label: 'Compare', icon: TrendingUp },
    { id: 'dataset' as const, label: 'Dataset', icon: BookOpen },
    { id: 'settings' as const, label: 'Settings', icon: Settings },
  ];



  return (
    <div className="flex h-screen bg-[#09090b] text-zinc-100 overflow-hidden" style={{ fontFamily: "'Inter', sans-serif" }}>
      {showLoader && (
        <div className={`fixed inset-0 z-50 bg-[#09090b] flex flex-col items-center justify-center transition-all duration-500 ease-in-out ${loaderFadingOut ? 'opacity-0 scale-105 pointer-events-none' : 'opacity-100 scale-100'
          }`}>
          <style>{`
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
            @keyframes fadeInUp {
              from {
                opacity: 0;
                transform: translateY(1px) scale(0.90);
              }
              to {
                opacity: 1;
                transform: translateY(0) scale(0.5);
              }
            }
            .animate-fade-in-up {
              animation: fadeInUp 0.2s cubic-bezier(0.12, 0.8, 0.2, 0.8) forwards;
            }
          `}</style>

          <div className="relative w-80 h-32 flex items-center justify-center overflow-hidden">
            {HELLOS.map((hello, idx) => (
              <span
                key={idx}
                className="absolute text-4xl sm:text-5xl font-medium text-zinc-100 tracking-tight drop-shadow-[0_0_35px_rgba(255,255,255,0.2)] pointer-events-none"
                style={{
                  fontFamily: "'Poppins', sans-serif",
                  transition: "opacity 200ms ease-out, transform 250ms cubic-bezier(0.16, 1, 0.3, 1), filter 250ms ease-out",
                  opacity: idx === loaderIndex ? 1 : 0,
                  transform: idx === loaderIndex
                    ? "translateY(0) scale(1)"
                    : idx < loaderIndex
                      ? "translateY(-24px) scale(0.92)"
                      : "translateY(24px) scale(0.92)",
                  filter: idx === loaderIndex ? "blur(0px)" : "blur(3px)"
                }}
              >
                {hello.text}
              </span>
            ))}
          </div>
        </div>
      )}

      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
          onClick={() => { setIsSidebarOpen(false); SoundEffects.playClick(); }}
        />
      )}

      {/* SIDEBAR */}
      <aside className={`w-60 fixed inset-y-0 left-0 z-40 md:relative md:flex flex-col border-r border-zinc-500/60 bg-[#09090b] transition-transform duration-200 ease-in-out ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}>
        {/* Logo */}
        <div className="h-15 flex items-center px-5 border-b border-zinc-800/60">
          <div className="flex items-center gap-2.5">
            {/* Box Container for Logo */}
            <div className="w-9 h-9 rounded-3xl bg-zinc-800 flex items-center justify-center overflow-hidden">
              <img
                src="public/logo.png"
                alt="Train Assistant Logo"
                className="w-full h-full object-contain"
              />
            </div>
            <span className="font-semibold text-lg text-zinc-200 tracking-tight">Train Assistant</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-3 px-3">
          {/* New Chat */}
          <button
            onClick={startNewChat}
            className="w-full flex items-center gap-2 px-3 py-2 mb-3 rounded-3xl text-sm font-medium text-zinc-300 border border-zinc-800 hover:bg-zinc-800/50 hover:text-zinc-100 transition-all duration-150"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New chat</span>
          </button>

          {/* Main nav */}
          <div className="space-y-0.5 mb-8">
            {navItems.map(item => (
              <button
                key={item.id}
                onClick={() => { setActiveView(item.id); setIsSidebarOpen(false); SoundEffects.playOpen(); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-3xl text-sm transition-all duration-150 ${activeView === item.id
                  ? 'bg-zinc-800/70 text-zinc-100 font-medium'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30'
                  }`}
              >
                <item.icon className="w-4 h-4" />
                <span>{item.label}</span>
              </button>
            ))}
          </div>

          {/* Conversations */}
          {conversations.length > 0 && (
            <div className="mb-8">
              <p className="text-[10px] font-bold text-zinc-200 uppercase tracking-wider px-3 mb-2">History</p>
              <div className="space-y-0.5">
                {conversations.map(chat => (
                  <div
                    key={chat.id}
                    onClick={() => selectChat(chat.id)}
                    className={`group w-full flex items-center justify-between px-3 py-1.5 rounded-3xl text-sm cursor-pointer transition-all duration-150 ${activeChatId === chat.id
                      ? 'bg-zinc-800/70 text-zinc-200'
                      : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30'
                      }`}
                  >
                    <span className="truncate text-[13px]">{chat.title}</span>
                    <button
                      onClick={(e) => deleteChat(e, chat.id)}
                      className="opacity-100 md:opacity-0 md:group-hover:opacity-100 p-0.5 hover:text-red-400 transition-all duration-150"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Uploads */}
          {uploads.length > 0 && (
            <div className="mb-8">
              <p className="text-[10px] font-bold text-zinc-200 uppercase tracking-wider px-3 mb-2">Uploads</p>
              <div className="space-y-0.5">
                {uploads.map(file => (
                  <div
                    key={file.id}
                    onClick={() => {
                      SoundEffects.playClick();
                      setActiveView('chat');
                      handleSendMessage(`Read and analyze the uploaded log file: ${file.filename}`);
                      setIsSidebarOpen(false);
                    }}
                    className="group flex items-center justify-between px-3 py-1.5 rounded-3xl text-[13px] text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30 cursor-pointer transition-all duration-150"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <FileText className="w-3.5 h-3.5 shrink-0 text-zinc-600" />
                      <span className="truncate">{file.filename}</span>
                    </div>
                    <button
                      onClick={(e) => deleteUpload(e, file.id)}
                      className="opacity-100 md:opacity-0 md:group-hover:opacity-100 p-0.5 hover:text-red-400 transition-all duration-150"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reports */}
          {reports.length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-zinc-200 uppercase tracking-wider px-3 mb-2">Reports</p>
              <div className="space-y-0.5">
                {reports.map(rep => (
                  <div
                    key={rep.id}
                    onClick={() => { previewReport(rep); setIsSidebarOpen(false); }}
                    className="group flex items-center justify-between px-3 py-1.5 rounded-2xl text-[13px] text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30 cursor-pointer truncate transition-all duration-150"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <FileText className="w-3.5 h-3.5 shrink-0 text-zinc-600" />
                      <span className="truncate">{rep.name}</span>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteReport(rep.id); }}
                      className="opacity-100 md:opacity-0 md:group-hover:opacity-100 p-0.5 hover:text-red-400 transition-all duration-150"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-zinc-800/60">
          <div className="flex items-center gap-2.5">
            {/* Box Container for Logo */}
            <div className="w-8 h-8 rounded-2xl bg-zinc-800 flex items-center justify-center overflow-hidden">
              <img
                src="public/logo.png"
                alt="Train Assistant Logo"
                className="w-full h-full object-contain"
              />
            </div>
            <div className="min-w-0">
              <p className="text-sm text-zinc-400 font-medium truncate">{settings.gemini_model}</p>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-15 flex items-center justify-between px-4 sm:px-6 border-b border-zinc-500/60 bg-[#09090b] shrink-0">
          <div className="flex items-center flex-1 min-w-0 mr-4">
            <button
              onClick={() => { setIsSidebarOpen(true); SoundEffects.playOpen(); }}
              className="p-2 mr-2 rounded-2xl text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 md:hidden"
            >
              <Menu className="w-5 h-5" />
            </button>
            <h2 className="text-lg sm:text-xl font-medium text-zinc-200 capitalize truncate">
              {activeView === 'chat' ? 'Chat' : activeView === 'gpu' ? 'GPU Monitor' : activeView}
            </h2>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <button
              onClick={() => {
                const newVal = !soundEnabled;
                setSoundEnabled(newVal);
                localStorage.setItem('sound_enabled', String(newVal));
                if (newVal) {
                  SoundEffects.setEnabled(true);
                  SoundEffects.playClick();
                }
              }}
              className="p-2.5 rounded-3xl text-zinc-500 border border-zinc-800 hover:bg-zinc-800/50 hover:text-zinc-200 transition-all duration-150"
              title={soundEnabled ? "Mute Sounds" : "Unmute Sounds"}
            >
              {soundEnabled ? (
                <Volume2 className="w-3.5 h-3.5 text-zinc-400" />
              ) : (
                <VolumeX className="w-3.5 h-3.5 text-zinc-600" />
              )}
            </button>

            <button
              onClick={() => { SoundEffects.playClick(); fileInputRef.current?.click(); }}
              className="flex items-center gap-1.5 sm:gap-2.5 px-2.5 sm:px-4 py-2 rounded-3xl text-xs font-medium text-zinc-400 border border-zinc-800 hover:bg-zinc-800/50 hover:text-zinc-200 transition-all duration-150"
            >
              <Upload className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Upload</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".csv,.log,.txt"
            />
            {activeView === 'gpu' && (
              <button
                onClick={() => { SoundEffects.playLoading(); fetchGPUInfo(); }}
                className="p-2.5 rounded-3xl text-zinc-500 border border-zinc-800 hover:bg-zinc-800/50 hover:text-zinc-200 transition-all duration-150"
              >
                <RefreshCw className={`w-2.5 h-2.5 ${gpuLoading ? 'animate-spin' : ''}`} />
              </button>
            )}
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">

          {/* DASHBOARD */}
          {activeView === 'dashboard' && (
            <div className="p-4 sm:p-6 lg:p-10 max-w-5xl mx-auto space-y-5">

              {/* Welcome & Instructions */}
              <div className="rounded-2xl border border-zinc-800/60 bg-gradient-to-br from-zinc-900/80 to-zinc-900/20 p-6 relative overflow-hidden">
                <div className="relative z-10 flex flex-col md:flex-row gap-6 md:gap-12">
                  {/* About */}
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2 text-zinc-200 mb-3">
                      <Cpu className="w-5 h-5 text-white-400" />
                      <h3 className="text-base font-semibold">About ML Train Assistant</h3>
                    </div>
                    <p className="text-sm text-zinc-400 leading-relaxed">
                      An intelligent dashboard for machine learning engineers to analyze model training logs. It combines
                      <strong> Google Gemini AI</strong> with specialized diagnostic tools to detect overfitting, recommend hyperparameter tweaks, and track GPU performance in real-time.
                    </p>
                  </div>

                  {/* How to Use */}
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2 text-zinc-200 mb-3">
                      <BookOpen className="w-5 h-5 text-white-400" />
                      <h3 className="text-base font-semibold">How to Use</h3>
                    </div>
                    <ul className="text-sm text-zinc-400 space-y-2">
                      <li className="flex gap-2"><span className="text-zinc-600">1.</span> <span><strong>Upload</strong> your training logs (CSV format) using the top bar.</span></li>
                      <li className="flex gap-2"><span className="text-zinc-600">2.</span> <span><strong>Chat</strong> with the AI agent to analyze metrics or debug issues.</span></li>
                      <li className="flex gap-2"><span className="text-zinc-600">3.</span> <span><strong>Monitor</strong> hardware usage in the GPU tab.</span></li>
                    </ul>
                  </div>

                  {/* Developer */}
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2 text-zinc-200 mb-3">
                      <GitBranch className="w-5 h-5 text-white-400" />
                      <h3 className="text-base font-semibold">Developer</h3>
                    </div>
                    <p className="text-sm text-zinc-400 leading-relaxed mb-3">
                      Build by <strong className="text-zinc-300">Arghadeep Pakhira</strong>.
                    </p>
                    <a
                      href="https://github.com/Argha2004"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition-colors duration-150 border border-zinc-700"
                    >
                      <GitBranch className="w-3.5 h-3.5" />
                      <span>View GitHub</span>
                    </a>
                  </div>
                </div>
              </div>

              {/* Stat cards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { label: 'Uploads', value: dashboardStats.num_uploads, sub: 'training logs' },
                  { label: 'Reports', value: dashboardStats.num_reports, sub: 'generated' },
                  { label: 'Avg AUC', value: dashboardStats.average_auc || '—', sub: 'validation' },
                  { label: 'GPUs', value: gpus.length, sub: gpus.length > 0 ? gpus[0].name : 'detecting...' },
                ].map((stat, i) => (
                  <div key={i} className="p-4 rounded-2xl border border-zinc-800/60 bg-zinc-900/30">
                    <p className="text-xs text-zinc-500 font-medium mb-1">{stat.label}</p>
                    <p className="text-2xl font-semibold text-zinc-100 tracking-tight">{stat.value}</p>
                    <p className="text-xs text-zinc-600 mt-1 truncate">{stat.sub}</p>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Experiments table */}
                <div className="col-span-1 lg:col-span-2 rounded-2xl border border-zinc-800/60 bg-zinc-900/30 overflow-hidden">
                  <div className="px-4 py-3 border-b border-zinc-800/40">
                    <h4 className="text-sm font-medium text-zinc-300">Recent Experiments</h4>
                  </div>
                  {dashboardStats.latest_experiments.length === 0 ? (
                    <div className="px-4 py-12 text-center text-zinc-600 text-sm">
                      No experiments uploaded yet.
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm min-w-[500px]">
                        <thead>
                          <tr className="text-xs text-zinc-500 border-b border-zinc-800/40">
                            <th className="text-left px-4 py-2.5 font-medium">Name</th>
                            <th className="text-center px-4 py-2.5 font-medium">Epochs</th>
                            <th className="text-center px-4 py-2.5 font-medium">Best AUC</th>
                            <th className="text-center px-4 py-2.5 font-medium">Val Loss</th>
                            <th className="text-right px-4 py-2.5 font-medium"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {dashboardStats.latest_experiments.map(run => (
                            <tr key={run.filename} className="border-b border-zinc-800/20 hover:bg-zinc-800/20 transition-colors duration-100">
                              <td className="px-4 py-3 text-zinc-300 truncate max-w-[200px]">{run.name}</td>
                              <td className="px-4 py-3 text-center text-zinc-400">{run.epochs}</td>
                              <td className="px-4 py-3 text-center text-zinc-300 font-mono text-xs">
                                {run.best_auc !== null ? run.best_auc.toFixed(4) : '—'}
                              </td>
                              <td className="px-4 py-3 text-center text-zinc-400 font-mono text-xs">
                                {run.best_val_loss !== null ? run.best_val_loss.toFixed(4) : '—'}
                              </td>
                              <td className="px-4 py-3 text-right">
                                <button
                                  onClick={() => {
                                    setActiveView('chat');
                                    handleSendMessage(`Analyze the run: ${run.filename}. Check for plateau and recommend batch size/scheduler.`);
                                  }}
                                  className="text-xs text-zinc-500 hover:text-zinc-200 transition-colors duration-150"
                                >
                                  Analyze →
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Recommendations */}
                <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/30 overflow-hidden">
                  <div className="px-4 py-3 border-b border-zinc-800/40">
                    <h4 className="text-sm font-medium text-zinc-300">Recommendations</h4>
                  </div>
                  <div className="p-3 space-y-2 max-h-[350px] overflow-y-auto">
                    {dashboardStats.recent_recommendations.map((rec, i) => (
                      <div key={i} className="p-3 rounded-2xl border border-zinc-800/40 bg-zinc-900/40">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className={`w-1.5 h-1.5 rounded-full ${rec.severity === 'high' ? 'bg-red-500' : rec.severity === 'moderate' ? 'bg-amber-500' : 'bg-emerald-500'
                            }`} />
                          <span className="text-xs font-medium text-zinc-400">{rec.metric}</span>
                        </div>
                        <p className="text-xs text-zinc-500 leading-relaxed">{rec.recommendation}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* CHAT */}
          {activeView === 'chat' && (
            <div className="flex flex-col h-full">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-6 py-6">
                <div className="max-w-4xl mx-auto space-y-4">
                  {messages.length === 0 && !isGenerating && (
                    <div className="text-center py-20">
                      <MessageSquare className="w-8 h-8 text-zinc-700 mx-auto mb-4" />
                      <h3 className="text-base font-medium text-zinc-400 mb-2">ML Diagnosis Assistant</h3>
                      <p className="text-sm text-zinc-600 max-w-md mx-auto">
                        Upload a training CSV log and ask about overfitting, batch sizing, or run comparisons.
                      </p>
                    </div>
                  )}

                  {messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] rounded-3xl px-4 py-3 ${msg.role === 'user'
                        ? 'bg-zinc-800 text-zinc-200'
                        : 'bg-zinc-900/50 border border-zinc-800/50 text-zinc-300'
                        }`}>
                        {msg.content && (
                          <div className="prose-custom text-sm">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                          </div>
                        )}

                        {msg.tool_calls && (
                          <div className="mt-2.5 pt-2.5 border-t border-zinc-700/30 space-y-1.5">
                            {msg.tool_calls.map((tc, idx) => (
                              <div key={idx} className="flex items-center gap-2 text-xs text-zinc-500 font-mono bg-zinc-800/40 px-2.5 py-1.5 rounded-3xl">
                                <CheckCircle2 className="w-3 h-3 text-zinc-600 shrink-0" />
                                <span className="truncate">{tc.name}()</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Render chart images from tool responses in history */}
                        {msg.tool_responses && (() => {
                          const charts = msg.tool_responses
                            .map(tr => tr.response)
                            .filter(r => r && (r.chart_url || r.result?.chart_url))
                            .map(r => r.chart_url || r.result?.chart_url);
                          if (charts.length === 0) return null;
                          return (
                            <div className="mt-3 space-y-3">
                              {charts.map((url: string, ci: number) => {
                                const fullUrl = `${API_BASE}${url}`;
                                const filename = url.split('/').pop() || 'chart.png';
                                return (
                                  <div key={ci} className="rounded-xl border border-zinc-800/40 overflow-hidden bg-zinc-900/40">
                                    <img
                                      src={fullUrl}
                                      alt={filename}
                                      className="w-full rounded-t-md"
                                      loading="lazy"
                                    />
                                    <div className="flex items-center justify-between px-3 py-2 border-t border-zinc-800/30">
                                      <span className="text-xs text-zinc-500 font-mono truncate">{filename}</span>
                                      <a
                                        href={fullUrl}
                                        download={filename}
                                        className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors duration-150"
                                      >
                                        <Download className="w-3 h-3" />
                                        <span>Download</span>
                                      </a>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  ))}

                  {/* Streaming text */}
                  {streamedText && (
                    <div className="flex justify-start">
                      <div className="max-w-[80%] rounded-3xl px-4 py-3 bg-zinc-900/50 border border-zinc-800/50 text-zinc-300">
                        <div className="prose-custom text-sm">
                          <ReactMarkdown>{streamedText}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Tool status + inline chart preview for completed chart tools */}
                  {currentTools.map((t, idx) => (
                    <div key={idx} className="flex justify-start">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2.5 px-3 py-2 rounded-2xl border border-zinc-700/40 bg-zinc-900/30 text-xs text-zinc-400" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                          {t.status === 'running' ? (
                            <Loader className="w-3.5 h-3.5 animate-spin text-zinc-500" />
                          ) : (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                          )}
                          <span>{t.name}</span>
                          <span className="text-zinc-600">
                            {t.status === 'running' ? 'running...' : 'done'}
                          </span>
                        </div>
                        {/* Show chart inline when tool completes and has a chart_url */}
                        {t.status === 'completed' && t.result && (t.result.chart_url || t.result.result?.chart_url) && (() => {
                          const chartUrl = t.result.chart_url || t.result.result?.chart_url;
                          const fullUrl = `${API_BASE}${chartUrl}`;
                          const filename = chartUrl.split('/').pop() || 'chart.png';
                          return (
                            <div className="rounded-3xl border border-zinc-800/40 overflow-hidden bg-zinc-900/40 max-w-md">
                              <img
                                src={fullUrl}
                                alt={filename}
                                className="w-full rounded-t-md"
                                loading="lazy"
                              />
                              <div className="flex items-center justify-between px-3 py-2 border-t border-zinc-800/30">
                                <span className="text-xs text-zinc-500 font-mono truncate">{filename}</span>
                                <a
                                  href={fullUrl}
                                  download={filename}
                                  className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors duration-150"
                                >
                                  <Download className="w-3 h-3" />
                                  <span>Download</span>
                                </a>
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  ))}

                  {/* Typing indicator */}
                  {isGenerating && !streamedText && currentTools.length === 0 && (
                    <div className="flex justify-start">
                      <div className="flex items-center gap-1.5 px-5 py-3 rounded-3xl bg-zinc-900/50 border border-zinc-800/40">
                        <div className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2.5 h-2.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>
              </div>

              {/* Input */}
              <div className="border-t border-zinc-800/60 px-4 sm:px-6 py-3 sm:py-4 bg-[#09090b]">
                <form
                  onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}
                  className="max-w-3xl mx-auto flex items-center gap-2 sm:gap-3"
                >
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 sm:p-4 rounded-full text-zinc-500 border border-zinc-800 hover:bg-zinc-500/50 hover:text-zinc-300 transition-all duration-150 shrink-0"
                  >
                    <Upload className="w-4 h-4" />
                  </button>
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder={isGenerating ? "Generating..." : "Ask about your training logs..."}
                    className="flex-1 bg-zinc-900/60 border border-zinc-800 rounded-full py-2.5 sm:py-4 px-4 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-600 transition-colors duration-150"
                    disabled={isGenerating}
                  />
                  <button
                    type="submit"
                    className="p-3 sm:p-4 rounded-full bg-zinc-200 text-zinc-900 hover:bg-white transition-all duration-150 disabled:opacity-30 shrink-0"
                    disabled={isGenerating || !chatInput.trim()}
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* GPU */}
          {activeView === 'gpu' && (
            <div className="p-4 sm:p-6 lg:p-10 max-w-4xl mx-auto space-y-4">
              {gpus.length === 0 ? (
                <div className="text-center py-20 text-zinc-600">
                  <Loader className="w-6 h-6 animate-spin mx-auto mb-3" />
                  <p className="text-sm">Querying GPU status...</p>
                </div>
              ) : (
                gpus.map(gpu => (
                  <div key={gpu.gpu_id} className="rounded-2xl border border-zinc-800/60 bg-zinc-900/30 p-5 space-y-5">
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0 pr-2">
                        <h4 className="text-sm font-medium text-zinc-200 truncate">{gpu.name}</h4>
                        <p className="text-xs text-zinc-600 mt-0.5">GPU {gpu.gpu_id}</p>
                      </div>
                      <div className="flex items-center gap-1.5 text-xs shrink-0">
                        <Thermometer className="w-3.5 h-3.5 text-zinc-500" />
                        <span className={`font-medium ${gpu.temperature_c > 75 ? 'text-red-400' : 'text-zinc-400'}`}>
                          {gpu.temperature_c}°C
                        </span>
                      </div>
                    </div>

                    {/* VRAM */}
                    <div>
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className="text-zinc-500">VRAM</span>
                        <span className="text-zinc-400 font-mono">{gpu.vram_used_gb}GB / {gpu.vram_total_gb}GB</span>
                      </div>
                      <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-zinc-400 rounded-full transition-all duration-500"
                          style={{ width: `${gpu.vram_utilization_pct}%` }}
                        />
                      </div>
                    </div>

                    {/* Core Utilization */}
                    <div>
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className="text-zinc-500">Core Utilization</span>
                        <span className="text-zinc-400 font-mono">{gpu.gpu_utilization_pct}%</span>
                      </div>
                      <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-zinc-500 rounded-full transition-all duration-500"
                          style={{ width: `${gpu.gpu_utilization_pct}%` }}
                        />
                      </div>
                    </div>

                    {/* Batch size hint */}
                    <div className="flex items-center justify-between p-4 rounded-xl bg-zinc-800/40 border border-zinc-800/40">
                      <div>
                        <p className="text-xs text-zinc-400 font-medium">Batch Size Multiplier</p>
                        <p className="text-[11px] text-zinc-600">Available headroom</p>
                      </div>
                      <span className="text-lg font-semibold text-zinc-300">+{gpu.estimated_batch_size_multiplier.toFixed(0)}x</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* COMPARISON */}
          {activeView === 'comparison' && (
            <div className="p-4 sm:p-6 lg:p-10 max-w-5xl mx-auto space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Selection */}
                <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/30 p-4 space-y-4">
                  <h4 className="text-sm font-medium text-zinc-300">Select Runs</h4>
                  {uploads.length === 0 ? (
                    <p className="text-xs text-zinc-600">Upload logs to get started.</p>
                  ) : (
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {uploads.map(file => (
                        <label key={file.id} className="flex items-center gap-1.5 cursor-pointer p-2.5 rounded-3xl hover:bg-zinc-800/30 transition-colors duration-100">
                          <input
                            type="checkbox"
                            checked={selectedRuns.includes(file.filename)}
                            onChange={() => toggleRunSelection(file.filename)}
                            className="w-3.5 h-3.5 accent-zinc-400 rounded"
                          />
                          <span className="text-sm text-zinc-400 truncate">{file.filename}</span>
                        </label>
                      ))}
                    </div>
                  )}

                  <button
                    onClick={runComparison}
                    disabled={selectedRuns.length === 0 || comparisonLoading}
                    className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-3xl text-sm font-medium bg-zinc-200 text-zinc-900 hover:bg-white disabled:opacity-30 transition-all duration-150"
                  >
                    {comparisonLoading ? (
                      <Loader className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <>
                        <TrendingUp className="w-3.5 h-3.5" />
                        <span>Compare</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Chart area */}
                <div className="col-span-1 lg:col-span-2 rounded-2xl border border-zinc-800/60 bg-zinc-900/30 p-4">
                  {comparisonData.length > 0 ? (
                    <div className="h-80 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={comparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                          <XAxis dataKey="epoch" stroke="#52525b" tick={{ fontSize: 11 }} />
                          <YAxis stroke="#52525b" tick={{ fontSize: 11 }} />
                          <ChartTooltip
                            contentStyle={{
                              backgroundColor: '#18181b',
                              borderColor: '#27272a',
                              borderRadius: '6px',
                              fontSize: '12px'
                            }}
                          />
                          <ChartLegend wrapperStyle={{ fontSize: '12px' }} />
                          <Line type="monotone" dataKey="run_a_loss" name="Run A" stroke="#a1a1aa" strokeWidth={1.5} dot={false} />
                          <Line type="monotone" dataKey="run_b_loss" name="Run B" stroke="#71717a" strokeWidth={1.5} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-80 flex flex-col items-center justify-center text-zinc-600">
                      <BarChart2 className="w-6 h-6 mb-2" />
                      <p className="text-sm">Select runs and click Compare.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* DATASET */}
          {activeView === 'dataset' && (() => {
            const COLORS = ['#d4d4d8', '#a1a1aa', '#71717a', '#52525b', '#3f3f46', '#27272a'];
            return (
              <div className="p-4 sm:p-6 lg:p-10 max-w-5xl mx-auto space-y-6">
                <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/30 p-5 space-y-6">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                      <h4 className="text-sm font-bold text-zinc-200 mb-1">Dataset Diagnostics</h4>
                      <p className="text-xs text-zinc-500">
                        Analyze class distributions, missing labels, and duplicate entries.
                      </p>
                    </div>
                    {uploads.length > 0 && (
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4">
                        <select
                          value={selectedDatasetFile}
                          onChange={(e) => {
                            setSelectedDatasetFile(e.target.value);
                            setDatasetAnalysisResult(null);
                          }}
                          className="bg-zinc-900 border border-zinc-800 rounded-3xl pl-3.5 pr-10 py-2.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-700 transition-colors"
                        >
                          <option value="">Select a file...</option>
                          {uploads.map(file => (
                            <option key={file.id} value={file.filename}>{file.filename}</option>
                          ))}
                        </select>

                        <button
                          disabled={!selectedDatasetFile || datasetAnalysisLoading}
                          onClick={() => handleAnalyzeDataset(selectedDatasetFile)}
                          className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-3xl text-xs font-medium bg-zinc-200 text-zinc-900 hover:bg-white disabled:opacity-40 disabled:hover:bg-zinc-200 transition-all duration-150"
                        >
                          {datasetAnalysisLoading ? (
                            <Loader className="w-3 h-3 animate-spin" />
                          ) : (
                            <Activity className="w-3 h-3" />
                          )}
                          Analyze
                        </button>
                      </div>
                    )}
                  </div>

                  {uploads.length === 0 ? (
                    <div className="text-center py-10 rounded-xl border border-dashed border-zinc-800 bg-zinc-900/20">
                      <BookOpen className="w-8 h-8 mx-auto mb-3 text-zinc-600" />
                      <p className="text-sm font-medium text-zinc-400">No datasets uploaded yet</p>
                      <p className="text-xs text-zinc-600 mt-1 max-w-xs mx-auto mb-4">
                        Upload a CSV log or dataset file from the sidebar to begin analysis.
                      </p>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-3xl text-xs font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-800 transition-all duration-150"
                      >
                        <Upload className="w-3 h-3" />
                        Upload CSV
                      </button>
                    </div>
                  ) : !selectedDatasetFile ? (
                    <div className="text-center py-10 rounded-xl border border-zinc-800/40 bg-zinc-950/20">
                      <BookOpen className="w-8 h-8 mx-auto mb-2.5 text-zinc-600" />
                      <p className="text-sm font-medium text-zinc-400">Select a dataset file to analyze</p>
                      <p className="text-xs text-zinc-600 mt-1">
                        Choose one of the uploaded CSV files from the dropdown menu above.
                      </p>
                    </div>
                  ) : datasetAnalysisLoading ? (
                    <div className="text-center py-16">
                      <Loader className="w-8 h-8 mx-auto mb-3 animate-spin text-zinc-500" />
                      <p className="text-sm text-zinc-400 font-medium">Running Diagnostics...</p>
                      <p className="text-xs text-zinc-600 mt-1">Analyzing shapes, class distribution, and data integrity</p>
                    </div>
                  ) : datasetAnalysisResult ? (
                    <div className="space-y-6">
                      {/* Basic Summary Cards */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="rounded-2xl border border-zinc-800/50 bg-zinc-950/30 p-3.5 space-y-1">
                          <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Total Records</span>
                          <p className="text-xl font-semibold text-zinc-200">
                            {datasetAnalysisResult.total_samples?.toLocaleString()}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-zinc-800/50 bg-zinc-950/30 p-3.5 space-y-1">
                          <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Duplicate Rows</span>
                          <p className={`text-xl font-semibold ${datasetAnalysisResult.duplicate_samples > 0 ? 'text-zinc-400' : 'text-zinc-200'}`}>
                            {datasetAnalysisResult.duplicate_samples?.toLocaleString()}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-zinc-800/50 bg-zinc-950/30 p-3.5 space-y-1">
                          <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Corrupted Files</span>
                          <p className={`text-xl font-semibold ${datasetAnalysisResult.corrupted_files_count > 0 ? 'text-red-400/80' : 'text-zinc-200'}`}>
                            {datasetAnalysisResult.corrupted_files_count}
                          </p>
                        </div>
                      </div>

                      {/* Chart Layout */}
                      {Object.keys(datasetAnalysisResult.class_distribution || {}).length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="h-60 rounded-2xl border border-zinc-800/40 bg-zinc-900/40 p-4 flex flex-col">
                            <span className="text-xs font-medium text-zinc-400 mb-3 block">Distribution Ratio</span>
                            <div className="flex-1 min-h-0">
                              <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                  <Pie
                                    data={Object.entries(datasetAnalysisResult.class_distribution).map(([name, value]) => ({
                                      name,
                                      value: Number(value)
                                    }))}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={45}
                                    outerRadius={65}
                                    paddingAngle={3}
                                    dataKey="value"
                                    strokeWidth={0}
                                  >
                                    {Object.keys(datasetAnalysisResult.class_distribution).map((_, index) => (
                                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                  </Pie>
                                  <ChartTooltip
                                    contentStyle={{
                                      backgroundColor: '#18181b',
                                      borderColor: '#27272a',
                                      borderRadius: '6px',
                                      fontSize: '12px'
                                    }}
                                  />
                                  <ChartLegend wrapperStyle={{ fontSize: '11px' }} />
                                </PieChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          <div className="h-60 rounded-2xl border border-zinc-800/40 bg-zinc-900/40 p-4 flex flex-col">
                            <span className="text-xs font-medium text-zinc-400 mb-3 block">Class Sample Counts</span>
                            <div className="flex-1 min-h-0">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                  data={Object.entries(datasetAnalysisResult.class_distribution).map(([name, count]) => ({
                                    name,
                                    count: Number(count)
                                  }))}
                                  margin={{ top: 10, right: 5, left: -20, bottom: 0 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                  <XAxis dataKey="name" stroke="#52525b" tick={{ fontSize: 11 }} />
                                  <YAxis stroke="#52525b" tick={{ fontSize: 11 }} />
                                  <ChartTooltip
                                    contentStyle={{
                                      backgroundColor: '#18181b',
                                      borderColor: '#27272a',
                                      borderRadius: '6px',
                                      fontSize: '12px'
                                    }}
                                  />
                                  <Bar dataKey="count" radius={[3, 3, 0, 0]}>

                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Diagnostics Details */}
                      <div className="rounded-2xl border border-zinc-800/40 bg-zinc-900/10 p-4 space-y-3">
                        <span className="text-sm font-bold text-zinc-400 block">Analysis Insights</span>
                        <p className="text-xs text-zinc-500 leading-relaxed">
                          {datasetAnalysisResult.message || `The dataset contains ${datasetAnalysisResult.total_samples} samples. We detected ${datasetAnalysisResult.duplicate_samples} duplicate items.`}
                        </p>

                        {/* Missing values checklist */}
                        {datasetAnalysisResult.missing_labels && Object.keys(datasetAnalysisResult.missing_labels).length > 0 && (
                          <div className="pt-3 border-t border-zinc-800/40">
                            <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block mb-2">Null / Missing Columns</span>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {Object.entries(datasetAnalysisResult.missing_labels).map(([col, val]: any) => (
                                <div key={col} className="flex justify-between items-center text-xs py-2 px-2 bg-zinc-900/30 rounded-lg border border-zinc-800/30">
                                  <span className="text-zinc-400 font-mono">{col}</span>
                                  <span className={val > 0 ? 'text-amber-500/80 font-medium' : 'text-zinc-600'}>{val} missing</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Corrupted Files list */}
                        {datasetAnalysisResult.corrupted_files && datasetAnalysisResult.corrupted_files.length > 0 && (
                          <div className="pt-3 border-t border-zinc-800/40">
                            <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider block mb-2">Corrupted Samples detected</span>
                            <div className="flex flex-wrap gap-1.5">
                              {datasetAnalysisResult.corrupted_files.map((file: string) => (
                                <span key={file} className="text-[11px] font-mono bg-red-950/20 text-red-400/80 border border-red-900/30 px-2 py-0.5 rounded truncate max-w-full">
                                  {file}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="text-center pt-2">
                        <button
                          onClick={() => {
                            setActiveView('chat');
                            handleSendMessage(`Analyze the dataset diagnostics for "${selectedDatasetFile}" in detail. Highlight any data imbalance, missing values, or potential data cleaning actions.`);
                          }}
                          className="inline-flex items-center gap-2 px-5 py-3 rounded-3xl text-sm font-medium bg-zinc-200 text-zinc-900 hover:bg-white transition-all duration-150"
                        >
                          <MessageSquare className="w-3.5 h-3.5" />
                          Discuss with AI Agent
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-10 rounded-md border border-zinc-800/40 bg-zinc-950/20">
                      <BookOpen className="w-8 h-8 mx-auto mb-2 text-zinc-600" />
                      <p className="text-sm font-medium text-zinc-400 font-semibold">Dataset Loaded</p>
                      <p className="text-xs text-zinc-600 mt-1 mb-4">
                        Click the "Analyze" button above or below to run diagnostic tools on {selectedDatasetFile}.
                      </p>
                      <button
                        onClick={() => handleAnalyzeDataset(selectedDatasetFile)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-zinc-250 text-zinc-900 hover:bg-white transition-all duration-150"
                      >
                        <Activity className="w-3 h-3" />
                        Run Analysis
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}

          {/* SETTINGS */}
          {activeView === 'settings' && (
            <div className="p-4 sm:p-6 lg:p-10 max-w-4xl mx-auto space-y-3">
              <div className="rounded-2xl border border-zinc-500/60 bg-zinc-900/30 p-5 space-y-5">
                {/* API Key */}
                <div>
                  <label className="block text-sm font-medium text-zinc-300 mb-1.5 flex items-center justify-between">
                    <span>Gemini API Key</span>
                    <a
                      href="https://aistudio.google.com/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[11px] text-zinc-500 hover:text-zinc-300 underline"
                    >
                      Get key from Google AI Studio
                    </a>
                  </label>
                  <input
                    type="password"
                    value={settings.gemini_api_key || ''}
                    onChange={(e) => saveSettings({ gemini_api_key: e.target.value })}
                    placeholder={settings.gemini_api_key ? "••••••••••••••••••••••••••••••••••••" : "AIzaSy..."}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-3xl p-2.5 text-sm text-zinc-300 focus:outline-none focus:border-zinc-600 transition-colors duration-150"
                  />
                  <p className="text-[11px] text-zinc-600 mt-1">
                    Your key is saved locally. It must start with <code className="text-zinc-500 bg-zinc-950 px-1 py-0.5 rounded">AIzaSy</code> or <code className="text-zinc-500 bg-zinc-950 px-1 py-0.5 rounded">AQ.</code>.
                  </p>
                </div>

                {/* Model */}
                <div>
                  <label className="block text-sm font-medium text-zinc-300 mb-1.5">Model</label>
                  <select
                    value={settings.gemini_model}
                    onChange={(e) => {
                      SoundEffects.playClick();
                      saveSettings({ gemini_model: e.target.value });
                    }}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-3xl p-2.5 text-sm text-zinc-300 focus:outline-none focus:border-zinc-600 transition-colors duration-150"
                  >
                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                    <option value="gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                    <option value="gemini-3.5-flash">Gemini 3.5 Flash</option>
                    <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash Lite</option>
                    <option value="gemma-4-31b-it">Gemma 4 31B IT</option>
                  </select>
                  <p className="text-[11px] text-zinc-600 mt-1">Free tier: Only models with available quota will work. Check your <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer" className="text-zinc-500 hover:text-zinc-300 underline">API rate limits</a> to verify.</p>
                </div>

                {/* System prompt */}
                <div>
                  <label className="block text-sm font-medium text-zinc-300 mb-1.5">System Instructions</label>
                  <textarea
                    rows={3}
                    value={settings.system_prompt}
                    onChange={(e) => saveSettings({ system_prompt: e.target.value })}
                    placeholder="Custom instructions for the AI agent..."
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-2xl p-2.5 text-sm text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-600 transition-colors duration-150 resize-none"
                  />
                </div>

                {/* Directories */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-zinc-500 mb-1.5">Logs Folder</label>
                    <input
                      type="text"
                      value={settings.upload_dir}
                      onChange={(e) => saveSettings({ upload_dir: e.target.value })}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-3xl p-2.5 text-sm text-zinc-300 focus:outline-none focus:border-zinc-600 transition-colors duration-150"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-zinc-500 mb-1.5">Reports Folder</label>
                    <input
                      type="text"
                      value={settings.report_dir}
                      onChange={(e) => saveSettings({ report_dir: e.target.value })}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-3xl p-2.5 text-sm text-zinc-300 focus:outline-none focus:border-zinc-600 transition-colors duration-150"
                    />
                  </div>
                </div>

                {/* Toggles */}
                <div className="pt-3 border-t border-zinc-800/40 space-y-3">
                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.streaming_toggle}
                      onChange={(e) => {
                        SoundEffects.playClick();
                        saveSettings({ streaming_toggle: e.target.checked });
                      }}
                      className="w-3.5 h-3.5 accent-zinc-400 rounded"
                    />
                    <span className="text-sm text-zinc-400">Enable SSE streaming</span>
                  </label>

                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.auto_save}
                      onChange={(e) => {
                        SoundEffects.playClick();
                        saveSettings({ auto_save: e.target.checked });
                      }}
                      className="w-3.5 h-3.5 accent-zinc-400 rounded"
                    />
                    <span className="text-sm text-zinc-400">Auto-save chat history</span>
                  </label>

                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={soundEnabled}
                      onChange={(e) => {
                        const newVal = e.target.checked;
                        setSoundEnabled(newVal);
                        localStorage.setItem('sound_enabled', String(newVal));
                        if (newVal) {
                          SoundEffects.setEnabled(true);
                          SoundEffects.playClick();
                        }
                      }}
                      className="w-3.5 h-3.5 accent-zinc-400 rounded"
                    />
                    <span className="text-sm text-zinc-400">Enable UI sound effects</span>
                  </label>
                </div>
              </div>

              {/* About Section */}
              <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/30 p-5 space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-zinc-200">About ML Train Assistant</h4>
                  <p className="text-xs text-zinc-500 mt-1.5 leading-relaxed">
                    An AI-powered ML training analysis platform that combines Google Gemini with Model Context Protocol (MCP) diagnostic tools to analyze training runs, detect issues, and provide actionable recommendations.
                  </p>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-zinc-800/40">
                  <div className="text-xs text-zinc-500">
                    Version 0.1.2
                  </div>
                  <a
                    href="https://github.com/Argha2004"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-3 py-1.5 rounded-2xl text-xs font-medium bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white transition-colors duration-150"
                  >
                    <GitBranch className="w-4 h-4" />
                    <span>Developer GitHub</span>
                  </a>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Report Preview Modal */}
        {previewReportUrl && (
          <div className="absolute inset-0 bg-black/70 z-50 flex items-center justify-center p-4 sm:p-8">
            <div className="bg-[#0a0a0c] border border-zinc-800 rounded-2xl w-full max-w-3xl h-5/6 flex flex-col overflow-hidden">
              <div className="px-4 sm:px-5 py-4 border-b border-zinc-800/60 flex items-center justify-between gap-3 min-w-0">
                <h3 className="text-sm font-medium text-zinc-200 truncate mr-2">{previewReportName}</h3>
                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={previewReportUrl}
                    download
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-3xl text-xs font-medium bg-zinc-200 text-zinc-900 hover:bg-white transition-all duration-150"
                  >
                    <Download className="w-3 h-3" />
                    <span className="hidden sm:inline">Download</span>
                  </a>
                  <button
                    onClick={() => setPreviewReportUrl(null)}
                    className="p-1.5 rounded-2xl text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50 transition-all duration-150"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-6">
                <div className="prose-custom text-sm max-w-none">
                  <ReactMarkdown>{previewReportContent}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        )
        }
      </main >
    </div >
  );
}
