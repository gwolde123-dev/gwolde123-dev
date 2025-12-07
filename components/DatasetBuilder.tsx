
import React, { useState, useMemo, useEffect } from 'react';
import { AisleNode, AisleLink, ModelConfig } from '../types';
import { CSV_BRAILLE_BLIND, CSV_CONTENT, CSV_PROGRESS, CSV_STUDENTS, AI_ALGORITHMS } from '../constants';
import { Download, FileJson, FileSpreadsheet, Database, Copy, Check, Terminal, Share2, Server, Filter, Shield, AlertTriangle, Wand2, RefreshCw, FileText, Code2, Layers, Beaker, Settings2, PlayCircle } from 'lucide-react';

interface DatasetBuilderProps {
  nodes: AisleNode[];
  links: AisleLink[];
}

type DatasetFormat = 'jsonl' | 'csv' | 'topology' | 'python_pytorch' | 'python_tf';
type DataSource = 'graph' | 'external';
type ExternalDatasetType = 'braille' | 'content' | 'progress' | 'students' | 'merged_interactions';

export const DatasetBuilder: React.FC<DatasetBuilderProps> = ({ nodes, links }) => {
  const [dataSource, setDataSource] = useState<DataSource>('graph');
  const [activeFormat, setActiveFormat] = useState<DatasetFormat>('jsonl');
  const [activeExternal, setActiveExternal] = useState<ExternalDatasetType>('braille');
  const [copystate, setCopyState] = useState(false);

  // --- PREPROCESSING STATE ---
  const [config, setConfig] = useState({
      removeNulls: true,
      normalizeText: false, // lowercase, trim
      anonymizePII: false, // mask known PII fields
      enrichContext: true, // add system prompts
      validateSchema: true, // filter incomplete records
      oneHotEncode: true, // for categorical features in external data
      normalizeNumerics: true // min-max scaling for scores/age
  });
  
  // --- AI MODEL CONFIGURATION (LABORATORY) ---
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
      taskType: 'classification',
      algorithm: 'mlp',
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      validationProtocol: 'holdout',
      kFoldSplits: 5,
      metrics: ['Accuracy', 'Loss']
  });

  const [healthStats, setHealthStats] = useState({
      originalRows: 0,
      cleanRows: 0,
      dropped: 0,
      piiMasked: 0,
      nullsFixed: 0
  });

  // --- CSV PARSER HELPER ---
  const parseCSV = (csvText: string) => {
      const lines = csvText.trim().split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      
      return lines.slice(1).map(line => {
          // Handle quoted strings
          const values: string[] = [];
          let currentVal = '';
          let inQuotes = false;
          
          for (let i = 0; i < line.length; i++) {
              const char = line[i];
              if (char === '"') {
                  inQuotes = !inQuotes;
              } else if (char === ',' && !inQuotes) {
                  values.push(currentVal.trim());
                  currentVal = '';
              } else {
                  currentVal += char;
              }
          }
          values.push(currentVal.trim());

          const obj: any = {};
          headers.forEach((h, i) => {
              obj[h] = values[i];
          });
          return obj;
      });
  };

  // --- DATA PROCESSING ENGINE ---
  const processData = () => {
      let stats = { originalRows: 0, cleanRows: 0, dropped: 0, piiMasked: 0, nullsFixed: 0 };
      
      // >>> BRANCH 1: GRAPH DATA
      if (dataSource === 'graph') {
          // ... (Existing Graph Logic - Keeping simplified for brevity, focusing on External)
          if (activeFormat === 'jsonl') {
              stats.originalRows = nodes.length;
              const cleaned = nodes.map(node => {
                  let desc = node.description || "";
                  if (config.normalizeText) desc = desc.toLowerCase().trim();
                  return JSON.stringify({ prompt: `Explain ${node.label}`, completion: desc });
              });
              stats.cleanRows = cleaned.length;
              setHealthStats(stats);
              return cleaned.join('\n');
          }
          return JSON.stringify(nodes, null, 2);
      } 
      
      // >>> BRANCH 2: EXTERNAL DATASETS
      else if (dataSource === 'external') {
          let rawData: any[] = [];
          
          if (activeExternal === 'braille') rawData = parseCSV(CSV_BRAILLE_BLIND);
          else if (activeExternal === 'content') rawData = parseCSV(CSV_CONTENT);
          else if (activeExternal === 'progress') rawData = parseCSV(CSV_PROGRESS);
          else if (activeExternal === 'students') rawData = parseCSV(CSV_STUDENTS);
          else if (activeExternal === 'merged_interactions') {
              const students = parseCSV(CSV_STUDENTS);
              const progress = parseCSV(CSV_PROGRESS);
              const studentMap = new Map(students.map(s => [s.student_id, s]));
              rawData = progress.map(p => ({ ...p, ...studentMap.get(p.student_id) || {} }));
          }

          stats.originalRows = rawData.length;

          // -- PREPROCESSING --
          let processed = rawData.map(row => {
              // 1. Remove Nulls
              if (config.removeNulls && Object.values(row).some(v => v === '')) { stats.dropped++; return null; }
              // 2. Normalize Text
              if (config.normalizeText) Object.keys(row).forEach(k => { if (typeof row[k] === 'string') row[k] = row[k].toLowerCase(); });
              // 3. One-Hot Encoding Simulation
              if (config.oneHotEncode && row['Gender']) row['gender_code'] = row['Gender'] === 'Male' ? 0 : 1;
              // 4. Normalize Numerics
              if (config.normalizeNumerics && row['score']) row['norm_score'] = (parseFloat(row['score']) / 100).toFixed(2);

              return row;
          }).filter(Boolean);

          stats.cleanRows = processed.length;
          setHealthStats(stats);

          // -- FORMAT OUTPUT --
          if (activeFormat === 'python_pytorch') return generatePyTorchCode(activeExternal, processed);
          if (activeFormat === 'python_tf') return generateTFCode(activeExternal, processed);

          // CSV Dump
          if (processed.length === 0) return "No data.";
          const keys = Object.keys(processed[0]);
          return [keys.join(','), ...processed.map((r: any) => keys.map(k => r[k]).join(','))].join('\n');
      }
      return "";
  };

  const generatePyTorchCode = (datasetName: string, sampleData: any[]) => {
      const featureCols = sampleData.length > 0 ? Object.keys(sampleData[0]).filter(k => k.includes('_code') || k.includes('norm_')) : ['feat1'];
      
      return `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
${modelConfig.validationProtocol === 'k-fold' ? 'from sklearn.model_selection import KFold' : 'from sklearn.model_selection import train_test_split'}
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = ${modelConfig.batchSize}
EPOCHS = ${modelConfig.epochs}
LEARNING_RATE = ${modelConfig.learningRate}
ALGORITHM = "${modelConfig.algorithm}" # ${AI_ALGORITHMS.find(a => a.id === modelConfig.algorithm)?.name}

class ${datasetName.charAt(0).toUpperCase() + datasetName.slice(1)}Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

# Load Data
df = pd.read_csv('aisle_${datasetName}.csv')
X = df[['${featureCols.join("', '")}']].values
y = df['score'].values if 'score' in df.columns else np.zeros(len(df))

# --- MODEL ARCHITECTURE ---
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        ${modelConfig.algorithm === 'lstm' ? 'self.lstm = nn.LSTM(64, 32, batch_first=True)' : ''}
        self.relu = nn.ReLU()
        self.output = nn.Linear(${modelConfig.algorithm === 'lstm' ? '32' : '64'}, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        ${modelConfig.algorithm === 'lstm' ? 'x, _ = self.lstm(x.unsqueeze(1))\nx = x[:, -1, :]' : ''}
        return self.output(x)

# --- TRAINING LOOP (${modelConfig.validationProtocol.toUpperCase()}) ---
${modelConfig.validationProtocol === 'k-fold' ? `
kf = KFold(n_splits=${modelConfig.kFoldSplits}, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{modelConfig.kFoldSplits}")
    train_ds = ${datasetName.charAt(0).toUpperCase() + datasetName.slice(1)}Dataset(X[train_idx], y[train_idx])
    val_ds = ${datasetName.charAt(0).toUpperCase() + datasetName.slice(1)}Dataset(X[val_idx], y[val_idx])
` : `
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
train_ds = ${datasetName.charAt(0).toUpperCase() + datasetName.slice(1)}Dataset(X_train, y_train)
val_ds = ${datasetName.charAt(0).toUpperCase() + datasetName.slice(1)}Dataset(X_val, y_val)
`}
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        # Training logic here...
        pass
`;
  };

  const generateTFCode = (datasetName: string, sampleData: any[]) => {
      const featureCols = sampleData.length > 0 ? Object.keys(sampleData[0]).filter(k => k.includes('_code') || k.includes('norm_')) : ['feat1'];
      return `import tensorflow as tf
from sklearn.model_selection import ${modelConfig.validationProtocol === 'k-fold' ? 'KFold' : 'train_test_split'}
import pandas as pd

# Config
BATCH_SIZE = ${modelConfig.batchSize}
EPOCHS = ${modelConfig.epochs}
LR = ${modelConfig.learningRate}

# Load Data
df = pd.read_csv('aisle_${datasetName}.csv')
X = df[['${featureCols.join("', '")}']].values
y = df['score'].values

# Model Definition (${modelConfig.algorithm.toUpperCase()})
def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)))
    ${modelConfig.algorithm === 'cnn' ? 'model.add(tf.keras.layers.Conv1D(32, 3, activation="relu"))' : ''}
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse', metrics=['mae'])
    return model

${modelConfig.validationProtocol === 'k-fold' ? `
kf = KFold(n_splits=${modelConfig.kFoldSplits}, shuffle=True)
for train_index, val_index in kf.split(X):
    model = create_model()
    model.fit(X[train_index], y[train_index], epochs=EPOCHS, batch_size=BATCH_SIZE)
` : `
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = create_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)
`}
`;
  };

  const processedData = useMemo(() => processData(), [nodes, links, dataSource, activeFormat, activeExternal, config, modelConfig]);

  const handleCopy = () => {
    navigator.clipboard.writeText(processedData);
    setCopyState(true);
    setTimeout(() => setCopyState(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto p-6 h-full flex flex-col">
      <div className="mb-6 flex justify-between items-start">
        <div>
            <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-3">
            <Server size={32} className="text-indigo-600" />
            Dataset Engine & Model Lab
            </h2>
            <p className="text-slate-600 mt-2">Data Preparation and AI Model Prototyping.</p>
        </div>
        <div className="flex bg-slate-100 p-1 rounded-lg">
            <button onClick={() => setDataSource('graph')} className={`px-4 py-2 text-xs font-bold uppercase rounded-md transition-all ${dataSource === 'graph' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500'}`}>Graph</button>
            <button onClick={() => { setDataSource('external'); setActiveFormat('csv'); }} className={`px-4 py-2 text-xs font-bold uppercase rounded-md transition-all ${dataSource === 'external' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500'}`}>External Datasets</button>
        </div>
      </div>

      <div className="flex-1 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden flex flex-col md:flex-row">
          
          {/* Sidebar Controls */}
          <div className="w-full md:w-80 bg-slate-50 border-r border-slate-200 flex flex-col overflow-y-auto">
              
              {/* Dataset Selection */}
              {dataSource === 'external' && (
                  <div className="p-4 border-b border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-3 flex items-center gap-2"><Database size={12}/> Dataset</div>
                      <select className="w-full p-2 text-sm border rounded bg-white" value={activeExternal} onChange={(e) => setActiveExternal(e.target.value as any)}>
                          {['braille', 'content', 'progress', 'students', 'merged_interactions'].map(ds => <option key={ds} value={ds}>{ds.replace('_', ' ').toUpperCase()}</option>)}
                      </select>
                  </div>
              )}

              {/* Model Laboratory (Only for Python Code) */}
              {activeFormat.includes('python') && (
                  <div className="p-4 border-b border-slate-200 bg-indigo-50/50">
                      <div className="text-xs font-bold text-indigo-700 uppercase mb-3 flex items-center gap-2"><Beaker size={12}/> AI Model Laboratory</div>
                      <div className="space-y-4">
                          <div>
                              <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Algorithm</label>
                              <select className="w-full p-2 text-xs border rounded bg-white" value={modelConfig.algorithm} onChange={e => setModelConfig({...modelConfig, algorithm: e.target.value})}>
                                  {AI_ALGORITHMS.map(alg => <option key={alg.id} value={alg.id}>{alg.name}</option>)}
                              </select>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-2">
                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Epochs</label>
                                  <input type="number" value={modelConfig.epochs} onChange={e => setModelConfig({...modelConfig, epochs: parseInt(e.target.value)})} className="w-full p-1.5 text-xs border rounded"/>
                              </div>
                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Batch</label>
                                  <input type="number" value={modelConfig.batchSize} onChange={e => setModelConfig({...modelConfig, batchSize: parseInt(e.target.value)})} className="w-full p-1.5 text-xs border rounded"/>
                              </div>
                          </div>

                          <div>
                              <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Validation Protocol</label>
                              <div className="flex gap-2">
                                  <button onClick={() => setModelConfig({...modelConfig, validationProtocol: 'holdout'})} className={`flex-1 py-1.5 text-[10px] rounded border ${modelConfig.validationProtocol === 'holdout' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600'}`}>Holdout</button>
                                  <button onClick={() => setModelConfig({...modelConfig, validationProtocol: 'k-fold'})} className={`flex-1 py-1.5 text-[10px] rounded border ${modelConfig.validationProtocol === 'k-fold' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600'}`}>K-Fold</button>
                              </div>
                          </div>
                          {modelConfig.validationProtocol === 'k-fold' && (
                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Splits (K)</label>
                                  <input type="range" min="2" max="10" value={modelConfig.kFoldSplits} onChange={e => setModelConfig({...modelConfig, kFoldSplits: parseInt(e.target.value)})} className="w-full h-1 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"/>
                                  <div className="text-right text-[10px] text-indigo-600 font-bold">{modelConfig.kFoldSplits}</div>
                              </div>
                          )}
                      </div>
                  </div>
              )}

              {/* Format Output */}
              <div className="p-4 border-b border-slate-200">
                  <div className="text-xs font-bold text-slate-400 uppercase mb-3">Output Format</div>
                  <select className="w-full p-2 text-sm border rounded bg-white" value={activeFormat} onChange={(e) => setActiveFormat(e.target.value as any)}>
                      <option value="csv">Cleaned CSV</option>
                      <option value="python_pytorch">PyTorch Training Script</option>
                      <option value="python_tf">TensorFlow Training Script</option>
                      {dataSource === 'graph' && <option value="jsonl">JSONL (Fine-Tuning)</option>}
                  </select>
              </div>

              <div className="p-4 flex-1">
                   <div className="text-xs font-bold text-slate-400 uppercase mb-3 flex items-center gap-2"><Wand2 size={12}/> Preprocessing</div>
                   <div className="space-y-2">
                       <label className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" checked={config.removeNulls} onChange={e => setConfig({...config, removeNulls: e.target.checked})}/> Remove Nulls</label>
                       <label className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" checked={config.normalizeText} onChange={e => setConfig({...config, normalizeText: e.target.checked})}/> Normalize Text</label>
                       <label className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" checked={config.oneHotEncode} onChange={e => setConfig({...config, oneHotEncode: e.target.checked})}/> One-Hot Encode</label>
                       <label className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" checked={config.normalizeNumerics} onChange={e => setConfig({...config, normalizeNumerics: e.target.checked})}/> Min-Max Scale</label>
                   </div>
                   <div className="mt-4 p-3 bg-slate-100 rounded text-[10px] space-y-1">
                       <div className="flex justify-between"><span>Input Rows:</span> <b>{healthStats.originalRows}</b></div>
                       <div className="flex justify-between text-emerald-600"><span>Clean Rows:</span> <b>{healthStats.cleanRows}</b></div>
                       <div className="flex justify-between text-red-500"><span>Dropped:</span> <b>{healthStats.dropped}</b></div>
                   </div>
              </div>
          </div>

          {/* Preview Panel */}
          <div className="flex-1 flex flex-col min-h-[400px]">
              <div className="p-3 border-b border-slate-100 flex justify-between items-center bg-white">
                  <div className="flex items-center gap-2 text-sm font-mono text-slate-500">
                      <Code2 size={14} /> <span>Generated Output</span>
                  </div>
                  <button onClick={handleCopy} className="text-xs flex items-center gap-1 text-indigo-600 font-medium hover:bg-indigo-50 px-2 py-1 rounded transition-colors">
                      {copystate ? "Copied" : "Copy"} <Copy size={14} />
                  </button>
              </div>
              <div className="flex-1 bg-slate-900 overflow-hidden relative p-4 overflow-y-auto custom-scrollbar">
                  <pre className={`font-mono text-xs whitespace-pre-wrap leading-relaxed ${activeFormat.includes('python') ? 'text-emerald-400' : 'text-slate-300'}`}>
                      {processedData}
                  </pre>
              </div>
          </div>
      </div>
    </div>
  );
};
