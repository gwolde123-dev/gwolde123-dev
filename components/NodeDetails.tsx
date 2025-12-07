
import React, { useState, useEffect, useMemo } from 'react';
import { AISLE_LINKS } from '../constants';
import { AisleNode, DataQualityMetrics } from '../types';
import { X, ArrowRight, ArrowLeft, Target, BookOpen, Database, Target as GoalIcon, Edit2, Check, X as XIcon, AlertCircle, Plus, Trash2, ShieldCheck, Calendar, AlertTriangle, FileText, Activity, Save, Sliders, Gauge, Code, Terminal, Cpu, Globe, Server, Box, Lock, Layout } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface NodeDetailsProps {
  node: AisleNode;
  onClose: () => void;
  onUpdateNode: (node: AisleNode) => void;
  isEditing: boolean;
  onToggleEdit: (isEditing: boolean) => void;
  simulationMode?: boolean;
}

// Simple syntax highlighting helper (visual only)
const CodeBlock = ({ code, lang }: { code: string; lang: string }) => (
    <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-700 shadow-sm text-left my-2">
        <div className="bg-slate-800 px-4 py-1.5 border-b border-slate-700 flex justify-between items-center">
            <span className="text-[10px] font-mono font-bold text-slate-400 uppercase">{lang}</span>
            <div className="flex gap-1.5">
                <div className="w-2 h-2 rounded-full bg-red-500/50"></div>
                <div className="w-2 h-2 rounded-full bg-yellow-500/50"></div>
                <div className="w-2 h-2 rounded-full bg-green-500/50"></div>
            </div>
        </div>
        <pre className="p-4 overflow-x-auto text-xs font-mono leading-relaxed text-slate-300">
            <code>{code}</code>
        </pre>
    </div>
);

export const NodeDetails: React.FC<NodeDetailsProps> = ({ node, onClose, onUpdateNode, isEditing, onToggleEdit, simulationMode }) => {
  const [activeTab, setActiveTab] = useState<'details' | 'research' | 'quality' | 'code' | 'simulation'>('details');
  const [editDescription, setEditDescription] = useState(node.description);
  
  // State for adding new objectives
  const [isAddingObjective, setIsAddingObjective] = useState(false);
  const [newObjCode, setNewObjCode] = useState('');
  const [newObjDesc, setNewObjDesc] = useState('');

  // State for adding new questions
  const [isAddingQuestion, setIsAddingQuestion] = useState(false);
  const [newQuestionCode, setNewQuestionCode] = useState('');
  const [newQuestionDesc, setNewQuestionDesc] = useState('');

  // State for managing datasets (Add & Edit)
  const [isAddingDataset, setIsAddingDataset] = useState(false);
  const [editingDatasetCode, setEditingDatasetCode] = useState<string | null>(null);
  const [newDatasetCode, setNewDatasetCode] = useState('');
  const [newDatasetDesc, setNewDatasetDesc] = useState('');
  const [newDatasetSource, setNewDatasetSource] = useState('');
  const [newDatasetLastUpdated, setNewDatasetLastUpdated] = useState('');
  const [newDatasetIssues, setNewDatasetIssues] = useState('');
  const [datasetError, setDatasetError] = useState<string | null>(null);

  // State for Data Quality editing
  const [editingQualityId, setEditingQualityId] = useState<string | null>(null);
  const [qualityForm, setQualityForm] = useState<DataQualityMetrics>({
      accuracy: '', accuracyBaseline: '', accuracyThreshold: '',
      completeness: '', completenessBaseline: '', completenessThreshold: '',
      timeliness: '', timelinessBaseline: '', timelinessThreshold: '',
      source: '', lastUpdated: '', knownIssues: ''
  });

  // Code Generation State
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  // Sync state when node changes
  useEffect(() => {
    setEditDescription(node.description);
    if (simulationMode) {
        setActiveTab('simulation');
    } else if (activeTab === 'simulation') {
        setActiveTab('details');
    }

    if (activeTab === 'details') {
        setIsAddingObjective(false);
        setNewObjCode('');
        setNewObjDesc('');
        setIsAddingQuestion(false);
        setNewQuestionCode('');
        setNewQuestionDesc('');
        setIsAddingDataset(false);
        setEditingDatasetCode(null);
        resetDatasetForm();
        setEditingQualityId(null);
    }
    // Reset selected file for code tab
    setSelectedFile(null);
  }, [node, activeTab, simulationMode]);
  
  // Find relationships
  const outgoing = AISLE_LINKS.filter(l => l.source === node.id);
  const incoming = AISLE_LINKS.filter(l => l.target === node.id);

  // --- DYNAMIC CODE GENERATOR ---
  const generatedCode = useMemo(() => {
      const snippets: { name: string; lang: string; type: 'frontend' | 'backend' | 'data' | 'infra' | 'security'; content: string }[] = [];
      const nodeName = node.id.replace(/[^a-zA-Z0-9]/g, '');
      const features = node.features || {};
      
       // 1. FRONTEND SNIPPETS (Main View + Widget)
      if (node.group === 'input' || node.group === 'output' || node.stakeholders?.includes('Student') || node.stakeholders?.includes('Teacher')) {
          snippets.push({
              name: `${nodeName}View.tsx`,
              lang: 'typescript',
              type: 'frontend',
              content: `import React, { useEffect, useState } from 'react';
import { View, Text, ${node.group === 'input' ? 'TextInput, Button, StyleSheet' : 'StyleSheet'} } from 'react-native';
${features.Platform ? `// Integrating ${features.Platform}` : ''}

interface ${nodeName}Props {
  userId: string;
  onInteraction: (data: any) => void;
}

export const ${nodeName}View: React.FC<${nodeName}Props> = ({ userId, onInteraction }) => {
  const [data, setData] = useState<any>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'active'>('idle');

  useEffect(() => {
    // Initialize ${node.label} module connection
    console.log("Mounting ${node.label}...");
    setStatus('loading');
    
    ${features.Connectivity?.includes('WebSocket') ? `
    // Real-time connection for ${features.Connectivity}
    const ws = new WebSocket('${process.env.WS_ENDPOINT || 'wss://api.aisle.edu'}/${nodeName.toLowerCase()}');
    ws.onmessage = (e) => {
        setData(JSON.parse(e.data));
        setStatus('active');
    };
    return () => ws.close();` : `
    // Mock initialization
    setTimeout(() => setStatus('active'), 500);
    `}
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>${node.fullLabel || node.label}</Text>
        <View style={[styles.statusBadge, status === 'active' ? styles.active : styles.idle]} />
      </View>
      
      <Text style={styles.desc}>${node.description}</Text>
      
      {/* Dynamic Feature Rendering */}
      ${node.group === 'input' ? `<View style={styles.inputContainer}>
        <Text style={styles.label}>Input Parameter:</Text>
        <TextInput style={styles.input} placeholder="Enter ${node.label} Data" />
        <Button title="Submit Data" onPress={() => onInteraction({ type: '${nodeName}', timestamp: Date.now() })} />
      </View>` : ''}
      
      ${node.group === 'output' ? `<View style={styles.metricContainer}>
        <Text style={styles.metricLabel}>${node.simulation?.label || 'Current Metric'}</Text>
        <Text style={styles.metricValue}>{data?.value || 0}</Text>
      </View>` : ''}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { padding: 20, flex: 1, backgroundColor: '#fff' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#1e293b' },
  desc: { fontSize: 14, color: '#64748b', marginBottom: 20, lineHeight: 20 },
  statusBadge: { width: 10, height: 10, borderRadius: 5 },
  active: { backgroundColor: '#10b981' },
  idle: { backgroundColor: '#94a3b8' },
  metricValue: { fontSize: 36, color: '${node.group === 'output' ? '#e11d48' : '#4f46e5'}', fontWeight: '800' }
});`
          });

          // Dashboard Widget
          snippets.push({
            name: `${nodeName}Widget.tsx`,
            lang: 'typescript',
            type: 'frontend',
            content: `// Reusable Dashboard Widget for ${node.fullLabel}
import React from 'react';
import { Card, ProgressBar } from '@aisle/ui-kit';

export const ${nodeName}Widget = ({ data, isLoading }) => {
  if (isLoading) return <Card.Skeleton />;

  return (
    <Card title="${node.fullLabel}" icon="${nodeName}Icon">
      <div className="flex justify-between items-end mb-4">
        <div className="flex flex-col">
           <span className="text-2xl font-bold">${node.group === 'output' ? (node.simulation?.value || 0) : 'Active'}</span>
           <span className="text-xs text-slate-500 uppercase">${node.simulation?.label || 'Status'}</span>
        </div>
        <span className="text-xs text-slate-400">Updated: Live</span>
      </div>
      
      {/* Visual Indicator */}
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] text-slate-400">
          <span>Target</span>
          <span>100%</span>
        </div>
        <ProgressBar 
          value={${node.group === 'output' ? 'data?.value || 0' : '100'}} 
          max={100} 
          color="${node.group === 'output' ? 'rose' : 'emerald'}" 
        />
      </div>
    </Card>
  );
};`
          });
      }

      // 2. BACKEND / AI SERVICES
      if (node.group === 'process' || features['AI Model'] || features.Processing) {
          snippets.push({
              name: `${nodeName.toLowerCase()}_service.py`,
              lang: 'python',
              type: 'backend',
              content: `from fastapi import FastAPI, WebSocket, Depends, HTTPException
import asyncio
${features['AI Model']?.includes('TF') ? 'import tensorflow as tf' : ''}
${features.Processing?.includes('Librosa') ? 'import librosa\nimport numpy as np' : ''}
from core.security import verify_token, check_permissions

app = FastAPI()

# Configuration
MODULE_NAME = "${node.fullLabel}"
${features['AI Model'] ? `MODEL_PATH = "assets/models/${nodeName.toLowerCase()}_v1.tflite"` : ''}

@app.on_event("startup")
async def load_resources():
    print(f"Loading {MODULE_NAME} resources...")
    ${features['AI Model'] ? '# Load TFLite interpreter\n    # interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)' : ''}

@app.websocket("/ws/${nodeName.toLowerCase()}")
async def websocket_endpoint(websocket: WebSocket, token: str = Depends(verify_token)):
    """
    Secure WebSocket endpoint for real-time ${node.category} processing.
    """
    await websocket.accept()
    
    # Check granular permissions
    if not check_permissions(token, "${nodeName.toLowerCase()}:stream"):
        await websocket.close(code=4003)
        return

    try:
        while True:
            # Receive raw input
            data = await websocket.receive_bytes()
            
            # Processing Logic: ${features.Processing || features['AI Model'] || 'Standard Logic'}
            ${features.Processing?.includes('Librosa') ? `
            # Example: Extract Mel Spectrogram
            # audio = np.frombuffer(data, dtype=np.float32)
            # mels = librosa.feature.melspectrogram(y=audio, sr=16000)
            result = {"status": "processed", "features": "extracted"}
            ` : 'result = {"status": "received", "size": len(data)}'}
            
            # Send result downstream to ${outgoing[0]?.target || 'Next Node'}
            await websocket.send_json(result)
    except Exception as e:
        print(f"Error in {MODULE_NAME}: {e}")`
          });
      }

      // 3. SQL MODEL (Data)
      if (features.Database || features.Storage) {
          snippets.push({
              name: `schema.sql`,
              lang: 'sql',
              type: 'data',
              content: `-- PostgreSQL Schema for ${node.fullLabel}
-- Optimized for Scalability & RLS Security

CREATE TABLE IF NOT EXISTS ${nodeName.toLowerCase()}_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Feature Specific Columns
    ${features.Database?.includes('Logs') ? 'log_level VARCHAR(20) CHECK (log_level IN (\'INFO\', \'WARN\', \'ERROR\')),\n    payload JSONB,' : ''}
    ${features.Database?.includes('Scores') ? 'score DECIMAL(5,2) CHECK (score >= 0 AND score <= 100),\n    metrics JSONB,' : ''}
    ${features['AI Model'] ? 'inference_latency_ms INTEGER,\n    model_version VARCHAR(50),' : ''}
    
    -- Metadata
    source VARCHAR(100),
    is_synced BOOLEAN DEFAULT FALSE
);

-- SCALABILITY: Indexes for high-frequency queries
CREATE INDEX IF NOT EXISTS idx_${nodeName.toLowerCase()}_user_time 
ON ${nodeName.toLowerCase()}_data (user_id, created_at DESC);

${features.Database?.includes('JSON') ? `CREATE INDEX IF NOT EXISTS idx_${nodeName.toLowerCase()}_payload_gin 
ON ${nodeName.toLowerCase()}_data USING GIN (payload);` : ''}

-- SECURITY: Row Level Security (RLS)
ALTER TABLE ${nodeName.toLowerCase()}_data ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own data
CREATE POLICY "Users view own ${nodeName} data"
ON ${nodeName.toLowerCase()}_data
FOR SELECT
USING (auth.uid() = user_id);

-- Policy: Teachers can view their students' data
CREATE POLICY "Teachers view student ${nodeName} data"
ON ${nodeName.toLowerCase()}_data
FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM class_roster 
        WHERE teacher_id = auth.uid() 
        AND student_id = ${nodeName.toLowerCase()}_data.user_id
    )
);`
          });
      }

      // 4. SECURITY CONFIGURATION
      snippets.push({
        name: `security_policy.json`,
        lang: 'json',
        type: 'security',
        content: `{
  "module": "${nodeName}",
  "securityLevel": "high",
  "authentication": {
    "method": "OAuth2 + JWT",
    "issuer": "https://auth.aisle.project",
    "requiredScopes": ["${nodeName.toLowerCase()}:read", "${nodeName.toLowerCase()}:write"]
  },
  "authorization": {
    "rbac": {
      "student": ["read:own", "create:own"],
      "teacher": ["read:class", "analyze:class"],
      "admin": ["*"]
    }
  },
  "dataProtection": {
    "encryptionAtRest": true,
    "fieldLevelEncryption": [${features.Database?.includes('Biometric') ? '"face_vector", "raw_audio"' : '"pii_fields"'}],
    "anonymization": {
      "enabled": true,
      "technique": "k-anonymity"
    }
  },
  "rateLimiting": {
    "requestsPerMinute": 60,
    "burst": 10
  }
}`
      });

      // 5. INFRASTRUCTURE & SCALABILITY
      snippets.push({
          name: `k8s-deployment.yaml`,
          lang: 'yaml',
          type: 'infra',
          content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${nodeName.toLowerCase()}-service
  labels:
    app: aisle
    component: ${nodeName.toLowerCase()}
spec:
  replicas: 2 # Baseline for redundancy
  selector:
    matchLabels:
      component: ${nodeName.toLowerCase()}
  template:
    metadata:
      labels:
        component: ${nodeName.toLowerCase()}
    spec:
      containers:
      - name: api
        image: aisle/${nodeName.toLowerCase()}:v1.2.0
        ports:
          - containerPort: 8080
        resources:
          # SCALABILITY: Resource Requests & Limits
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
          - name: DB_POOL_SIZE
            value: "20"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
# Horizontal Pod Autoscaler (HPA) for Scalability
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${nodeName.toLowerCase()}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${nodeName.toLowerCase()}-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70`
      });

      return snippets;
  }, [node]);

  useEffect(() => {
      // Default to first snippet when tab opens
      if (activeTab === 'code' && generatedCode.length > 0 && !selectedFile) {
          setSelectedFile(generatedCode[0].name);
      }
  }, [activeTab, generatedCode, selectedFile]);

  const getGroupColor = (group: string) => {
      switch(group) {
          case 'input': return 'bg-blue-100 text-blue-800 border-blue-200';
          case 'process': return 'bg-green-100 text-green-800 border-green-200';
          case 'output': return 'bg-red-100 text-red-800 border-red-200';
          default: return 'bg-slate-100 text-slate-800 border-slate-200';
      }
  };

  const getGroupLabel = (group: string) => {
      switch(group) {
          case 'input': return 'Input (Foundation)';
          case 'process': return 'Intermediate (Process)';
          case 'output': return 'Output (Results)';
          default: return 'Context';
      }
  };

  const isValid = editDescription.trim().length > 0;

  const handleSaveDescription = () => {
    if (!isValid) return;
    onUpdateNode({ ...node, description: editDescription });
    onToggleEdit(false);
  };

  const handleCancelDescription = () => {
    setEditDescription(node.description);
    onToggleEdit(false);
  };

  const handleSimulationChange = (val: number) => {
      onUpdateNode({
          ...node,
          simulation: {
              ...node.simulation!,
              value: val
          }
      });
  };

  // --- Handlers for Research Tabs (Objectives, Questions, Datasets) ---
  const handleAddObjective = () => {
    if (!newObjCode.trim() || !newObjDesc.trim()) return;
    const currentResearch = node.research || {};
    const currentObjectives = currentResearch.objectives || [];
    const currentDescriptions = currentResearch.objectiveDescriptions || {};
    if (currentObjectives.includes(newObjCode.toUpperCase())) { alert("Objective code already exists."); return; }
    const updatedResearch = {
        ...currentResearch,
        objectives: [...currentObjectives, newObjCode.toUpperCase()],
        objectiveDescriptions: { ...currentDescriptions, [newObjCode.toUpperCase()]: newObjDesc }
    };
    onUpdateNode({ ...node, research: updatedResearch });
    setIsAddingObjective(false); setNewObjCode(''); setNewObjDesc('');
  };
  const handleDeleteObjective = (code: string) => {
    if (!node.research) return;
    const updated = (node.research.objectives || []).filter(c => c !== code);
    const descs = { ...node.research.objectiveDescriptions }; delete descs[code];
    onUpdateNode({ ...node, research: { ...node.research, objectives: updated, objectiveDescriptions: descs } });
  };
  const handleAddQuestion = () => {
    if (!newQuestionCode.trim() || !newQuestionDesc.trim()) return;
    const currentResearch = node.research || {};
    const currentQuestions = currentResearch.questions || [];
    const currentDescriptions = currentResearch.questionDescriptions || {};
    if (currentQuestions.includes(newQuestionCode.toUpperCase())) { alert("Question code already exists."); return; }
    const updatedResearch = {
        ...currentResearch,
        questions: [...currentQuestions, newQuestionCode.toUpperCase()],
        questionDescriptions: { ...currentDescriptions, [newQuestionCode.toUpperCase()]: newQuestionDesc }
    };
    onUpdateNode({ ...node, research: updatedResearch });
    setIsAddingQuestion(false); setNewQuestionCode(''); setNewQuestionDesc('');
  };
  const handleDeleteQuestion = (code: string) => {
    if (!node.research) return;
    const updated = (node.research.questions || []).filter(c => c !== code);
    const descs = { ...node.research.questionDescriptions }; delete descs[code];
    onUpdateNode({ ...node, research: { ...node.research, questions: updated, questionDescriptions: descs } });
  };
  
  const resetDatasetForm = () => {
    setNewDatasetCode(''); setNewDatasetDesc(''); setNewDatasetSource(''); setNewDatasetLastUpdated(''); setNewDatasetIssues(''); setDatasetError(null);
  };
  const handleAddDataset = () => {
    if (!newDatasetCode.trim() || !newDatasetDesc.trim()) { setDatasetError("Dataset code and description are required."); return; }
    const currentResearch = node.research || {};
    const currentDatasets = currentResearch.datasets || [];
    const currentDescriptions = currentResearch.datasetDescriptions || {};
    const code = newDatasetCode.toUpperCase();
    if (currentDatasets.includes(code)) { setDatasetError(`Dataset ${code} already exists.`); return; }
    const updatedResearch = {
        ...currentResearch,
        datasets: [...currentDatasets, code],
        datasetDescriptions: { ...currentDescriptions, [code]: newDatasetDesc }
    };
    const initialQuality: DataQualityMetrics = {
        source: newDatasetSource, lastUpdated: newDatasetLastUpdated, knownIssues: newDatasetIssues,
        accuracy: '', accuracyBaseline: '', accuracyThreshold: '',
        completeness: '', completenessBaseline: '', completenessThreshold: '',
        timeliness: '', timelinessBaseline: '', timelinessThreshold: ''
    };
    onUpdateNode({ ...node, research: updatedResearch, dataQuality: { ...node.dataQuality, [code]: initialQuality } });
    setIsAddingDataset(false); resetDatasetForm();
  };
  const handleStartEditDataset = (code: string) => {
      const desc = node.research?.datasetDescriptions?.[code] || '';
      const dq = node.dataQuality?.[code] || {};
      setNewDatasetCode(code); setNewDatasetDesc(desc); setNewDatasetSource(dq.source || ''); setNewDatasetLastUpdated(dq.lastUpdated || ''); setNewDatasetIssues(dq.knownIssues || ''); setDatasetError(null); setEditingDatasetCode(code); setIsAddingDataset(false);
  };
  const handleSaveEditDataset = () => {
    if (!newDatasetDesc.trim()) { setDatasetError("Description is required."); return; }
    const code = editingDatasetCode!;
    const updatedResearch = { ...node.research, datasetDescriptions: { ...node.research?.datasetDescriptions, [code]: newDatasetDesc } };
    const updatedQualityEntry = { ...node.dataQuality?.[code], source: newDatasetSource, lastUpdated: newDatasetLastUpdated, knownIssues: newDatasetIssues };
    onUpdateNode({ ...node, research: updatedResearch as any, dataQuality: { ...node.dataQuality, [code]: updatedQualityEntry } });
    setEditingDatasetCode(null); resetDatasetForm();
  };
  const handleDeleteDataset = (code: string) => {
    if (!node.research) return;
    const updated = (node.research.datasets || []).filter(c => c !== code);
    const descs = { ...node.research.datasetDescriptions }; delete descs[code];
    onUpdateNode({ ...node, research: { ...node.research, datasets: updated, datasetDescriptions: descs } });
  };
  
  const handleEditQuality = (dsCode: string) => {
      const existing = node.dataQuality?.[dsCode] || {
          accuracy: '', accuracyBaseline: '', accuracyThreshold: '',
          completeness: '', completenessBaseline: '', completenessThreshold: '',
          timeliness: '', timelinessBaseline: '', timelinessThreshold: '',
          source: '', lastUpdated: '', knownIssues: ''
      };
      setQualityForm({ ...existing }); setEditingQualityId(dsCode);
  };
  const handleSaveQuality = (dsCode: string) => {
      const updatedQuality = { ...node.dataQuality, [dsCode]: qualityForm };
      onUpdateNode({ ...node, dataQuality: updatedQuality }); setEditingQualityId(null);
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className={`p-6 border-b ${getGroupColor(node.group)} bg-opacity-30 flex justify-between items-start`}>
        <div>
           <div className="flex flex-wrap gap-2 mb-2">
             <div className={`inline-block px-2 py-1 rounded text-xs font-semibold uppercase tracking-wide border ${getGroupColor(node.group)}`}>
               {getGroupLabel(node.group)}
             </div>
             {node.conceptualRole && (
               <div className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-bold uppercase tracking-wide border bg-indigo-100 text-indigo-900 border-indigo-200">
                 <Target size={10} />
                 {node.conceptualRole}
               </div>
             )}
           </div>
           <h2 className="text-2xl font-bold text-slate-900 leading-tight">{node.fullLabel || node.label}</h2>
           <p className="text-slate-600 font-medium mt-1">{node.category}</p>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-black/5 rounded-full transition-colors flex-shrink-0 ml-2">
            <X size={20} className="text-slate-500" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 px-6 overflow-x-auto">
        <button onClick={() => setActiveTab('details')} className={`py-3 px-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${activeTab === 'details' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>Details</button>
        <button onClick={() => setActiveTab('code')} className={`py-3 px-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap flex items-center gap-1 ${activeTab === 'code' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}><Code size={14} /> Implementation</button>
        <button onClick={() => setActiveTab('research')} className={`py-3 px-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${activeTab === 'research' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>Research</button>
        <button onClick={() => setActiveTab('quality')} className={`py-3 px-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${activeTab === 'quality' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>Quality</button>
        {simulationMode && (
            <button onClick={() => setActiveTab('simulation')} className={`py-3 px-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap flex items-center gap-1 ${activeTab === 'simulation' ? 'border-rose-600 text-rose-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}><Sliders size={14} /> Test Model</button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        
        {activeTab === 'details' && (
            <>
                {/* Description */}
                <section>
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Description</h3>
                        {!isEditing ? (
                             <button onClick={() => onToggleEdit(true)} className="text-slate-400 hover:text-indigo-600 transition-colors p-1 rounded hover:bg-slate-100" title="Edit Description"><Edit2 size={14} /></button>
                        ) : (
                            <div className="flex gap-2">
                                <button onClick={handleSaveDescription} disabled={!isValid} className={`p-1 rounded transition-colors ${isValid ? 'text-emerald-600 hover:bg-emerald-50' : 'text-slate-300 cursor-not-allowed'}`}><Check size={16} /></button>
                                <button onClick={handleCancelDescription} className="text-red-500 hover:bg-red-50 p-1 rounded"><XIcon size={16} /></button>
                            </div>
                        )}
                    </div>
                    {isEditing ? (
                        <>
                            <textarea value={editDescription} onChange={(e) => setEditDescription(e.target.value)} className={`w-full p-3 border rounded-lg focus:ring-2 outline-none text-slate-800 text-lg leading-relaxed min-h-[150px] ${isValid ? 'border-indigo-200 focus:ring-indigo-500 focus:border-indigo-500' : 'border-red-300 focus:ring-red-200 focus:border-red-400'}`} autoFocus placeholder="Description cannot be empty..." />
                            {!isValid && <p className="text-xs text-red-500 mt-2 font-medium flex items-center gap-1 animate-pulse"><AlertCircle size={12} /><span>Description cannot be empty.</span></p>}
                        </>
                    ) : (
                        <p className="text-slate-800 leading-relaxed text-lg">{node.description}</p>
                    )}
                </section>

                {/* Data Features & Bindings */}
                {node.features && Object.keys(node.features).length > 0 && (
                    <section className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2"><Activity size={14} /> Data Features & Bindings</h3>
                        <div className="space-y-2">
                            {Object.entries(node.features).map(([key, value], idx) => (
                                <div key={idx} className="flex justify-between items-center text-sm border-b border-slate-100 last:border-0 pb-2 last:pb-0">
                                    <span className="font-semibold text-slate-700">{key}</span>
                                    <span className="text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded text-xs border border-indigo-100">{value}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Connectivity */}
                <div className="grid grid-cols-1 gap-6">
                    {incoming.length > 0 && (
                        <section>
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2"><ArrowLeft size={14} /> Inputs (Influenced By)</h3>
                            <div className="flex flex-wrap gap-2">
                                {incoming.map((link, i) => (
                                    <div key={i} className="px-3 py-1.5 bg-slate-100 rounded text-sm text-slate-600 border border-slate-200">
                                        <span className="font-semibold text-slate-800">{link.source}</span>
                                        <span className="mx-1 text-slate-400 italic">via</span>
                                        <span className="text-xs uppercase bg-white px-1 border rounded">{link.type}</span>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}
                    {outgoing.length > 0 && (
                        <section>
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">Outputs (Leads To) <ArrowRight size={14} /></h3>
                            <div className="flex flex-wrap gap-2">
                                {outgoing.map((link, i) => (
                                    <div key={i} className="px-3 py-1.5 bg-slate-100 rounded text-sm text-slate-600 border border-slate-200">
                                        <span className="font-semibold text-slate-800">{link.target}</span>
                                        <span className="mx-1 text-slate-400 italic">via</span>
                                        <span className="text-xs uppercase bg-white px-1 border rounded">{link.type}</span>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}
                </div>
            </>
        )}

        {/* --- IMPLEMENTATION / CODE TAB --- */}
        {activeTab === 'code' && (
            <div className="space-y-6 animate-in fade-in duration-300">
                <div className="bg-slate-900 text-slate-300 p-4 rounded-lg border border-slate-700">
                    <p className="text-xs flex gap-2 items-center">
                        <Terminal size={16} className="text-emerald-400" />
                        Dynamic Implementation Generator (v1.3)
                    </p>
                    <p className="text-[10px] mt-1 text-slate-500 pl-6">
                        Auto-generated stack (UI, SQL, Security, Infra) based on node features: {Object.keys(node.features || {}).join(', ')}
                    </p>
                </div>

                <div className="flex flex-col md:flex-row gap-4 h-[500px]">
                    {/* File Explorer Sidebar */}
                    <div className="w-full md:w-1/3 bg-slate-50 rounded-lg border border-slate-200 overflow-hidden flex flex-col">
                        <div className="bg-slate-100 px-3 py-2 border-b border-slate-200 text-[10px] font-bold uppercase text-slate-500">Project Structure</div>
                        <div className="p-2 space-y-1 overflow-y-auto flex-1">
                            {generatedCode.map(file => (
                                <button
                                    key={file.name}
                                    onClick={() => setSelectedFile(file.name)}
                                    className={`w-full text-left px-3 py-2 rounded text-xs flex items-center gap-2 transition-colors ${selectedFile === file.name ? 'bg-indigo-100 text-indigo-700 font-semibold' : 'text-slate-600 hover:bg-slate-100'}`}
                                >
                                    {file.type === 'frontend' && <Globe size={14} className="text-blue-500" />}
                                    {file.type === 'backend' && <Cpu size={14} className="text-yellow-600" />}
                                    {file.type === 'data' && <Database size={14} className="text-emerald-500" />}
                                    {file.type === 'infra' && <Box size={14} className="text-slate-500" />}
                                    {file.type === 'security' && <Lock size={14} className="text-red-500" />}
                                    {file.name}
                                </button>
                            ))}
                            {generatedCode.length === 0 && (
                                <p className="text-xs text-slate-400 p-2 italic">No code templates available for this node type.</p>
                            )}
                        </div>
                    </div>

                    {/* Code Viewer */}
                    <div className="w-full md:w-2/3 flex flex-col">
                         {selectedFile ? (
                             generatedCode.filter(f => f.name === selectedFile).map(file => (
                                 <div key={file.name} className="h-full">
                                     <CodeBlock code={file.content} lang={file.lang} />
                                     <div className="text-[10px] text-slate-400 mt-1 flex justify-end gap-4">
                                         <span>Lines: {file.content.split('\n').length}</span>
                                         <span>Type: {file.type.toUpperCase()}</span>
                                     </div>
                                 </div>
                             ))
                         ) : (
                             <div className="h-full border border-dashed border-slate-300 rounded-lg flex flex-col items-center justify-center text-slate-400">
                                 <Code size={32} className="mb-2 opacity-50" />
                                 <p className="text-sm">Select a file to view implementation details</p>
                             </div>
                         )}
                    </div>
                </div>
            </div>
        )}

        {/* --- SIMULATION TAB --- */}
        {activeTab === 'simulation' && simulationMode && (
            <div className="space-y-8 animate-in fade-in duration-300">
                <div className="bg-rose-50 p-4 rounded-lg border border-rose-100 mb-6">
                    <p className="text-xs text-rose-800 flex gap-2"><Sliders size={16} /> Test Modeling: Adjust input variables to simulate impacts.</p>
                </div>
                {node.simulation ? (
                    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold text-slate-900">{node.simulation.label || "Activity Level"}</h3>
                            <div className="text-2xl font-mono font-bold text-rose-600">{Math.round(node.simulation.value)}%</div>
                        </div>
                        {node.simulation.isFixed ? (
                             <div>
                                 <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Adjust Variable</label>
                                 <input type="range" min="0" max="100" value={node.simulation.value} onChange={(e) => handleSimulationChange(parseInt(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-rose-600" />
                                 <div className="flex justify-between text-xs text-slate-400 mt-2"><span>Low (0)</span><span>High (100)</span></div>
                                 <p className="text-xs text-slate-500 mt-4 italic">Independent variable impacting downstream nodes.</p>
                             </div>
                        ) : (
                            <div>
                                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Calculated Outcome</label>
                                <div className="h-4 w-full bg-slate-100 rounded-full overflow-hidden border border-slate-200"><div className="h-full bg-gradient-to-r from-rose-400 to-emerald-500 transition-all duration-500 ease-out" style={{ width: `${node.simulation.value}%` }}></div></div>
                                <p className="text-xs text-slate-500 mt-4 italic">Dependent on inputs from "{incoming.map(l=>l.source).join(', ')}".</p>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center py-8 text-slate-400"><Gauge size={32} className="mx-auto mb-2 opacity-50" /><p>No simulation parameters defined.</p></div>
                )}
            </div>
        )}

        {/* Research Tab Content */}
        {activeTab === 'research' && (
            <div className="space-y-8 animate-in fade-in duration-300">
                {/* Objectives */}
                <div className="bg-indigo-50/50 rounded-xl p-5 border border-indigo-100">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2 text-indigo-700 font-bold uppercase text-xs tracking-wider"><GoalIcon size={16} /> Research Objectives</div>
                        {!isAddingObjective && <button onClick={() => setIsAddingObjective(true)} className="flex items-center gap-1 text-[10px] bg-indigo-100 text-indigo-700 px-2 py-1 rounded hover:bg-indigo-200 transition-colors"><Plus size={12} /> Add Objective</button>}
                    </div>
                    <div className="space-y-4">
                        {(node.research?.objectives || []).map((objCode, idx) => (
                            <div key={idx} className="bg-white p-3 rounded-lg border border-indigo-100 shadow-sm relative group">
                                <div className="flex justify-between items-start mb-1">
                                    <span className="inline-block px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-bold">{objCode}</span>
                                    <button onClick={() => handleDeleteObjective(objCode)} className="text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"><Trash2 size={12} /></button>
                                </div>
                                <div className="prose prose-sm max-w-none text-slate-700"><ReactMarkdown>{node.research!.objectiveDescriptions?.[objCode] || "Pending."}</ReactMarkdown></div>
                            </div>
                        ))}
                    </div>
                    {isAddingObjective && <div className="mt-4 bg-white p-4 rounded-lg border border-indigo-200 shadow-md"> <div className="flex flex-col gap-3"> <div><input type="text" value={newObjCode} onChange={(e) => setNewObjCode(e.target.value.toUpperCase())} placeholder="O#" className="w-full text-xs p-2 border rounded" autoFocus /></div> <div><textarea value={newObjDesc} onChange={(e) => setNewObjDesc(e.target.value)} placeholder="Description..." className="w-full text-xs p-2 border rounded" /></div> <div className="flex gap-2 justify-end"><button onClick={() => setIsAddingObjective(false)} className="text-xs px-3 py-1.5 text-slate-500 bg-slate-100 rounded">Cancel</button><button onClick={handleAddObjective} disabled={!newObjCode.trim()} className="text-xs px-3 py-1.5 bg-indigo-600 text-white rounded">Add</button></div></div></div>}
                </div>
                {/* Questions */}
                <div className="bg-amber-50/50 rounded-xl p-5 border border-amber-100">
                     <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2 text-amber-700 font-bold uppercase text-xs tracking-wider"><BookOpen size={16} /> Research Questions</div>
                        {!isAddingQuestion && <button onClick={() => setIsAddingQuestion(true)} className="flex items-center gap-1 text-[10px] bg-amber-100 text-amber-700 px-2 py-1 rounded hover:bg-amber-200 transition-colors"><Plus size={12} /> Add Question</button>}
                    </div>
                    <div className="space-y-4">
                         {(node.research?.questions || []).map((rq, idx) => (<div key={idx} className="bg-white p-3 rounded-lg border border-amber-100 shadow-sm relative group"><div className="flex justify-between items-start mb-1"><span className="inline-block px-2 py-0.5 bg-amber-100 text-amber-700 rounded text-xs font-bold mb-1">{rq}</span><button onClick={() => handleDeleteQuestion(rq)} className="text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"><Trash2 size={12} /></button></div><div className="text-sm text-slate-800 font-medium">{node.research!.questionDescriptions?.[rq] || "Description pending."}</div></div>))}
                    </div>
                    {isAddingQuestion && <div className="mt-4 bg-white p-4 rounded-lg border border-amber-200 shadow-md"><div className="flex flex-col gap-3"><div><input type="text" value={newQuestionCode} onChange={(e) => setNewQuestionCode(e.target.value.toUpperCase())} placeholder="RQ#" className="w-full text-xs p-2 border rounded" /></div><div><textarea value={newQuestionDesc} onChange={(e) => setNewQuestionDesc(e.target.value)} placeholder="Description..." className="w-full text-xs p-2 border rounded" /></div><div className="flex gap-2 justify-end"><button onClick={() => setIsAddingQuestion(false)} className="text-xs px-3 py-1.5 text-slate-500 bg-slate-100 rounded">Cancel</button><button onClick={handleAddQuestion} disabled={!newQuestionCode.trim()} className="text-xs px-3 py-1.5 bg-amber-600 text-white rounded">Add</button></div></div></div>}
                </div>
                {/* Datasets */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2 text-slate-600 font-bold uppercase text-xs tracking-wider">
                            <Database size={16} /> Associated Datasets
                        </div>
                        {!isAddingDataset && !editingDatasetCode && (
                            <button onClick={() => { setIsAddingDataset(true); resetDatasetForm(); }} className="flex items-center gap-1 text-[10px] bg-slate-200 text-slate-700 px-2 py-1 rounded hover:bg-slate-300 transition-colors">
                                <Plus size={12} /> Add Dataset
                            </button>
                        )}
                    </div>
                    
                    <div className="space-y-3">
                        {(node.research?.datasets || []).map((ds, idx) => {
                             if (editingDatasetCode === ds) { 
                                 return (
                                     <div key={ds} className="bg-white p-4 rounded-lg border border-indigo-300 shadow-md ring-1 ring-indigo-100 animate-in zoom-in-95 duration-200">
                                         <div className="flex flex-col gap-3">
                                             <div className="flex justify-between items-center mb-1">
                                                 <span className="font-bold text-indigo-700 text-xs bg-indigo-50 px-2 py-1 rounded border border-indigo-100">{ds}</span>
                                             </div>
                                             
                                             <div>
                                                 <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Description <span className="text-red-400">*</span></label>
                                                 <textarea value={newDatasetDesc} onChange={(e) => setNewDatasetDesc(e.target.value)} className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none min-h-[60px]" placeholder="Detailed description of the dataset..." />
                                             </div>

                                             <div className="grid grid-cols-2 gap-3">
                                                 <div>
                                                     <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Source</label>
                                                     <input type="text" value={newDatasetSource} onChange={(e) => setNewDatasetSource(e.target.value)} className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" placeholder="e.g. Survey" />
                                                 </div>
                                                 <div>
                                                     <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Last Updated</label>
                                                     <input type="date" value={newDatasetLastUpdated} onChange={(e) => setNewDatasetLastUpdated(e.target.value)} className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" />
                                                 </div>
                                             </div>

                                             <div>
                                                 <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Known Issues</label>
                                                 <textarea value={newDatasetIssues} onChange={(e) => setNewDatasetIssues(e.target.value)} className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" placeholder="Optional notes on data quality..." />
                                             </div>

                                             {datasetError && <div className="text-xs text-red-500 flex items-center gap-1"><AlertCircle size={12} /> {datasetError}</div>}
                                             
                                             <div className="flex gap-2 justify-end mt-2 pt-2 border-t border-slate-100">
                                                 <button onClick={() => { setEditingDatasetCode(null); resetDatasetForm(); }} className="text-xs px-3 py-1.5 text-slate-500 bg-slate-100 rounded hover:bg-slate-200">Cancel</button>
                                                 <button onClick={handleSaveEditDataset} className="text-xs px-3 py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-700 flex items-center gap-1"><Save size={12} /> Save Changes</button>
                                             </div>
                                         </div>
                                     </div>
                                 ); 
                             }
                             
                             const dq = node.dataQuality?.[ds] || {};
                             
                             return (
                                 <div key={idx} className="bg-white px-3 py-3 rounded-lg border border-slate-200 shadow-sm flex flex-col gap-2 group hover:border-indigo-200 transition-all">
                                      <div className="flex items-start justify-between">
                                         <div className="flex flex-col gap-1">
                                             <div className="flex items-center gap-2">
                                                 <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                                                 <span className="font-bold text-slate-800 text-sm">{ds}</span>
                                             </div>
                                             <p className="text-xs text-slate-600 pl-3.5 leading-relaxed">{node.research!.datasetDescriptions?.[ds] || "Description pending."}</p>
                                         </div>
                                         <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                             <button onClick={() => handleStartEditDataset(ds)} className="text-slate-400 hover:text-indigo-600 p-1.5 rounded hover:bg-indigo-50" title="Edit Dataset"><Edit2 size={14} /></button>
                                             <button onClick={() => handleDeleteDataset(ds)} className="text-slate-400 hover:text-red-500 p-1.5 rounded hover:bg-red-50" title="Delete Dataset"><Trash2 size={14} /></button>
                                         </div>
                                     </div>
                                     
                                     {(dq.source || dq.lastUpdated || dq.knownIssues) && (
                                         <div className="flex flex-wrap gap-2 text-[10px] pl-3.5 mt-1">
                                             {dq.source && <span className="bg-slate-50 text-slate-500 px-1.5 py-0.5 rounded border border-slate-100 flex items-center gap-1"><FileText size={10} /> {dq.source}</span>}
                                             {dq.lastUpdated && <span className="bg-slate-50 text-slate-500 px-1.5 py-0.5 rounded border border-slate-100 flex items-center gap-1"><Calendar size={10} /> {dq.lastUpdated}</span>}
                                             {dq.knownIssues && <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded border border-amber-100 flex items-center gap-1"><AlertTriangle size={10} /> Issues</span>}
                                         </div>
                                     )}
                                 </div>
                             );
                        })}
                        {(!node.research?.datasets || node.research.datasets.length === 0) && !isAddingDataset && (
                            <div className="text-center py-6 text-slate-400 text-xs italic bg-white rounded-lg border border-dashed border-slate-200">No datasets assigned yet.</div>
                        )}
                    </div>
                    
                    {isAddingDataset && (
                         <div className="mt-4 bg-white p-4 rounded-lg border border-slate-200 shadow-md ring-1 ring-slate-200 animate-in slide-in-from-top-2">
                            <div className="flex flex-col gap-3">
                                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide border-b border-slate-100 pb-2 mb-1">New Dataset Entry</h4>
                                
                                <div>
                                    <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Code <span className="text-red-400">*</span></label>
                                    <input type="text" value={newDatasetCode} onChange={(e) => { setNewDatasetCode(e.target.value.toUpperCase()); setDatasetError(null); }} placeholder="DS##" className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-slate-500 outline-none" autoFocus />
                                </div>
                                
                                <div>
                                    <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Description <span className="text-red-400">*</span></label>
                                    <textarea value={newDatasetDesc} onChange={(e) => { setNewDatasetDesc(e.target.value); setDatasetError(null); }} placeholder="Detailed description..." className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-slate-500 outline-none min-h-[60px]" />
                                </div>
                                
                                {/* Metadata Fields for Quick Add */}
                                <div className="grid grid-cols-2 gap-3 bg-slate-50 p-2 rounded border border-slate-100">
                                    <div>
                                        <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Source</label>
                                        <input type="text" value={newDatasetSource} onChange={(e) => setNewDatasetSource(e.target.value)} className="w-full text-xs p-1.5 border rounded" placeholder="Optional" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Last Updated</label>
                                        <input type="date" value={newDatasetLastUpdated} onChange={(e) => setNewDatasetLastUpdated(e.target.value)} className="w-full text-xs p-1.5 border rounded" />
                                    </div>
                                    <div className="col-span-2">
                                        <label className="text-[10px] font-bold text-slate-400 uppercase mb-1 block">Known Issues</label>
                                        <input type="text" value={newDatasetIssues} onChange={(e) => setNewDatasetIssues(e.target.value)} className="w-full text-xs p-1.5 border rounded" placeholder="Optional notes" />
                                    </div>
                                </div>

                                {datasetError && <div className="text-xs text-red-500 flex items-center gap-1 mt-1"><AlertCircle size={12} /> {datasetError}</div>}

                                <div className="flex gap-2 justify-end mt-2">
                                    <button onClick={() => { setIsAddingDataset(false); resetDatasetForm(); }} className="text-xs px-3 py-1.5 text-slate-500 bg-slate-100 rounded hover:bg-slate-200">Cancel</button>
                                    <button onClick={handleAddDataset} className="text-xs px-3 py-1.5 bg-slate-700 text-white rounded hover:bg-slate-800 flex items-center gap-1"><Plus size={12} /> Add Dataset</button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        )}

        {/* Quality Tab */}
        {activeTab === 'quality' && (
    <div className="space-y-6 animate-in fade-in duration-300">
        <div className="bg-blue-50/50 p-4 rounded-lg border border-blue-100 mb-6">
            <p className="text-xs text-blue-700 flex gap-2"><ShieldCheck size={16} /> Monitor data quality metrics, baselines, and metadata.</p>
        </div>
        {(!node.research?.datasets || node.research.datasets.length === 0) ? (
            <div className="text-center py-10 text-slate-400">
                <Database size={32} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">No datasets associated with this node.</p>
            </div>
        ) : (
            <div className="space-y-6">
                {node.research.datasets.map(ds => {
                    const isEditing = editingQualityId === ds;
                    const metrics = isEditing ? qualityForm : (node.dataQuality?.[ds] || {});

                    return (
                        <div key={ds} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden transition-all hover:shadow-md">
                            {/* Header */}
                            <div className="bg-slate-50 px-4 py-3 border-b border-slate-100 flex justify-between items-center">
                                <div className="flex items-center gap-2">
                                    <Database size={14} className="text-slate-400"/>
                                    <span className="font-bold text-sm text-slate-800">{ds}</span>
                                </div>
                                {!isEditing && (
                                    <button onClick={() => handleEditQuality(ds)} className="text-xs font-medium text-indigo-600 flex items-center gap-1 hover:text-indigo-800 transition-colors">
                                        <Edit2 size={12} /> Edit Metrics
                                    </button>
                                )}
                            </div>

                            {/* Body */}
                            <div className="p-4">
                                {isEditing ? (
                                    <div className="space-y-4">
                                        {/* Metric Groups */}
                                        {['accuracy', 'completeness', 'timeliness'].map((metric) => (
                                            <div key={metric} className="bg-slate-50 p-3 rounded border border-slate-100">
                                                <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2 border-b border-slate-200 pb-1">{metric}</h4>
                                                <div className="grid grid-cols-3 gap-2">
                                                    <div>
                                                        <label className="text-[9px] text-slate-400">Current</label>
                                                        <input 
                                                            value={(qualityForm as any)[metric] || ''} 
                                                            onChange={e => setQualityForm({...qualityForm, [metric]: e.target.value})} 
                                                            className="w-full text-xs p-1.5 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" 
                                                            placeholder="%"
                                                        />
                                                    </div>
                                                    <div>
                                                        <label className="text-[9px] text-slate-400">Baseline</label>
                                                        <input 
                                                            value={(qualityForm as any)[`${metric}Baseline`] || ''} 
                                                            onChange={e => setQualityForm({...qualityForm, [`${metric}Baseline`]: e.target.value})} 
                                                            className="w-full text-xs p-1.5 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" 
                                                            placeholder="Target"
                                                        />
                                                    </div>
                                                    <div>
                                                        <label className="text-[9px] text-slate-400">Threshold</label>
                                                        <input 
                                                            value={(qualityForm as any)[`${metric}Threshold`] || ''} 
                                                            onChange={e => setQualityForm({...qualityForm, [`${metric}Threshold`]: e.target.value})} 
                                                            className="w-full text-xs p-1.5 border rounded focus:ring-1 focus:ring-indigo-500 outline-none" 
                                                            placeholder="Min"
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                        ))}

                                        {/* Metadata */}
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Source</label>
                                                <input 
                                                    value={qualityForm.source || ''} 
                                                    onChange={e => setQualityForm({...qualityForm, source: e.target.value})} 
                                                    className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none"
                                                    placeholder="e.g., Student Survey"
                                                />
                                            </div>
                                            <div>
                                                <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Last Updated</label>
                                                <input 
                                                    type="date"
                                                    value={qualityForm.lastUpdated || ''} 
                                                    onChange={e => setQualityForm({...qualityForm, lastUpdated: e.target.value})} 
                                                    className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none"
                                                />
                                            </div>
                                            <div className="col-span-2">
                                                <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Known Issues</label>
                                                <textarea 
                                                    value={qualityForm.knownIssues || ''} 
                                                    onChange={e => setQualityForm({...qualityForm, knownIssues: e.target.value})} 
                                                    className="w-full text-xs p-2 border rounded focus:ring-1 focus:ring-indigo-500 outline-none min-h-[60px]"
                                                    placeholder="Any data gaps or latency issues..."
                                                />
                                            </div>
                                        </div>

                                        {/* Actions */}
                                        <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
                                            <button onClick={() => setEditingQualityId(null)} className="text-xs px-3 py-1.5 bg-slate-100 text-slate-600 rounded hover:bg-slate-200">Cancel</button>
                                            <button onClick={() => handleSaveQuality(ds)} className="text-xs px-3 py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-700 shadow-sm">Save Changes</button>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        {/* Metrics Display */}
                                        <div className="grid grid-cols-3 gap-4">
                                            {['Accuracy', 'Completeness', 'Timeliness'].map(label => {
                                                const key = label.toLowerCase() as keyof DataQualityMetrics;
                                                const val = metrics[key] as string;
                                                const baseline = metrics[`${key}Baseline` as keyof DataQualityMetrics] as string;
                                                const threshold = metrics[`${key}Threshold` as keyof DataQualityMetrics] as string;
                                                
                                                return (
                                                    <div key={label} className="bg-slate-50 p-2 rounded border border-slate-100 text-center">
                                                        <div className="text-[9px] font-bold text-slate-400 uppercase mb-1">{label}</div>
                                                        <div className={`text-sm font-bold ${val ? 'text-slate-800' : 'text-slate-300'}`}>{val || "—"}</div>
                                                        {(baseline || threshold) && (
                                                            <div className="mt-1 pt-1 border-t border-slate-200 flex justify-center gap-2 text-[9px] text-slate-500">
                                                                {baseline && <span title="Baseline">Base: {baseline}</span>}
                                                                {threshold && <span title="Threshold">Min: {threshold}</span>}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>

                                        {/* Metadata Display */}
                                        {(metrics.source || metrics.lastUpdated || metrics.knownIssues) && (
                                            <div className="text-xs space-y-2 border-t border-slate-100 pt-3">
                                                {metrics.source && (
                                                    <div className="flex gap-2">
                                                        <span className="font-bold text-slate-500 w-20">Source:</span>
                                                        <span className="text-slate-700">{metrics.source}</span>
                                                    </div>
                                                )}
                                                {metrics.lastUpdated && (
                                                    <div className="flex gap-2">
                                                        <span className="font-bold text-slate-500 w-20">Last Updated:</span>
                                                        <span className="text-slate-700">{metrics.lastUpdated}</span>
                                                    </div>
                                                )}
                                                {metrics.knownIssues && (
                                                    <div className="flex gap-2 items-start">
                                                        <span className="font-bold text-amber-600 w-20 flex-shrink-0 flex items-center gap-1"><AlertTriangle size={10}/> Issues:</span>
                                                        <span className="text-amber-800 bg-amber-50 px-2 py-1 rounded border border-amber-100 w-full">{metrics.knownIssues}</span>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                        
                                        {/* Empty State hint */}
                                        {!(metrics.source || metrics.lastUpdated || metrics.knownIssues || metrics.accuracy || metrics.completeness || metrics.timeliness) && (
                                            <div className="text-center text-xs text-slate-400 italic py-2">No quality metrics defined. Click edit to add.</div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        )}
    </div>
)}

      </div>
    </div>
  );
};
