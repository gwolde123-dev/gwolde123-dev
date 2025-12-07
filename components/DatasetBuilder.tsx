
import React, { useState, useMemo, useEffect } from 'react';
import { AisleNode, AisleLink, ModelConfig } from '../types';
import { CSV_BRAILLE_BLIND, CSV_CONTENT, CSV_PROGRESS, CSV_STUDENTS, CSV_EMOTION_MULTIMODAL, AI_ALGORITHMS } from '../constants';
import { Download, Database, Copy, Server, Wand2, FileText, Code2, Beaker, Settings2, Sliders, Cpu, Activity, BrainCircuit, Check } from 'lucide-react';

interface DatasetBuilderProps {
  nodes: AisleNode[];
  links: AisleLink[];
}

type DatasetFormat = 'jsonl' | 'csv' | 'python_pytorch' | 'python_tf';
type DataSource = 'graph' | 'external';
type ExternalDatasetType = 'braille' | 'content' | 'progress' | 'students' | 'merged_interactions' | 'emotion';

export const DatasetBuilder: React.FC<DatasetBuilderProps> = ({ nodes, links }) => {
  const [dataSource, setDataSource] = useState<DataSource>('graph');
  const [activeFormat, setActiveFormat] = useState<DatasetFormat>('jsonl');
  const [activeExternal, setActiveExternal] = useState<ExternalDatasetType>('braille');
  const [copystate, setCopyState] = useState(false);

  // --- PREPROCESSING STATE ---
  const [config, setConfig] = useState({
      removeNulls: true,
      normalizeText: false,
      anonymizePII: false,
      oneHotEncode: true,
      normalizeNumerics: true
  });
  
  // --- AI MODEL CONFIGURATION (LABORATORY) ---
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
      taskType: 'classification',
      algorithm: 'mlp',
      epochs: 20,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: 'adam',
      hiddenLayers: 2,
      layerSize: 64,
      dropout: 0.2,
      activation: 'relu',
      validationProtocol: 'holdout',
      kFoldSplits: 5,
      metrics: ['Accuracy', 'F1-Score'],
      transformerLayers: 2,
      attentionHeads: 4,
      embeddingDim: 64
  });

  const [healthStats, setHealthStats] = useState({
      originalRows: 0,
      cleanRows: 0,
      dropped: 0
  });

  // Smart Defaults based on Dataset
  useEffect(() => {
      if (dataSource === 'external') {
          if (activeExternal === 'progress') {
              setModelConfig(prev => ({ ...prev, taskType: 'forecasting', algorithm: 'lstm', epochs: 50, metrics: ['RMSE', 'MAE'] }));
          } else if (activeExternal === 'emotion') {
              setModelConfig(prev => ({ ...prev, taskType: 'classification', algorithm: 'cnn', epochs: 30, metrics: ['Accuracy', 'Precision'] }));
          } else if (activeExternal === 'braille') {
              setModelConfig(prev => ({ ...prev, taskType: 'classification', algorithm: 'rf', epochs: 10, metrics: ['Accuracy'] }));
          } else {
              setModelConfig(prev => ({ ...prev, taskType: 'classification', algorithm: 'mlp', metrics: ['Accuracy'] }));
          }
      }
  }, [activeExternal, dataSource]);

  // --- CSV PARSER HELPER ---
  const parseCSV = (csvText: string) => {
      const lines = csvText.trim().split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      
      return lines.slice(1).map(line => {
          const values: string[] = [];
          let currentVal = '';
          let inQuotes = false;
          for (let i = 0; i < line.length; i++) {
              const char = line[i];
              if (char === '"') { inQuotes = !inQuotes; } 
              else if (char === ',' && !inQuotes) { values.push(currentVal.trim()); currentVal = ''; } 
              else { currentVal += char; }
          }
          values.push(currentVal.trim());
          const obj: any = {};
          headers.forEach((h, i) => { obj[h] = values[i]; });
          return obj;
      });
  };

  // --- DATA PROCESSING ENGINE ---
  const processData = () => {
      let stats = { originalRows: 0, cleanRows: 0, dropped: 0 };
      
      if (dataSource === 'graph') {
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
      } else if (dataSource === 'external') {
          let rawData: any[] = [];
          
          if (activeExternal === 'braille') rawData = parseCSV(CSV_BRAILLE_BLIND);
          else if (activeExternal === 'content') rawData = parseCSV(CSV_CONTENT);
          else if (activeExternal === 'progress') rawData = parseCSV(CSV_PROGRESS);
          else if (activeExternal === 'students') rawData = parseCSV(CSV_STUDENTS);
          else if (activeExternal === 'emotion') rawData = parseCSV(CSV_EMOTION_MULTIMODAL);
          else if (activeExternal === 'merged_interactions') {
              const students = parseCSV(CSV_STUDENTS);
              const progress = parseCSV(CSV_PROGRESS);
              const studentMap = new Map(students.map(s => [s.student_id, s]));
              rawData = progress.map(p => ({ ...p, ...studentMap.get(p.student_id) || {} }));
          }

          stats.originalRows = rawData.length;

          // -- PREPROCESSING --
          let processed = rawData.map(row => {
              if (config.removeNulls && Object.values(row).some(v => v === '')) { stats.dropped++; return null; }
              if (config.normalizeText) Object.keys(row).forEach(k => { if (typeof row[k] === 'string') row[k] = row[k].toLowerCase(); });
              // Simple encoding simulation
              if (config.oneHotEncode) {
                  if (row['Gender']) row['gender_code'] = row['Gender'] === 'Male' ? 0 : 1;
                  if (row['Blindness_Type']) row['blindness_code'] = row['Blindness_Type'] === 'Congenital' ? 0 : 1;
              }
              if (config.normalizeNumerics) {
                  ['score', 'time_spent', 'resilience_score', 'efficacy_score'].forEach(key => {
                      if (row[key]) row[`norm_${key}`] = parseFloat(row[key]) > 1 ? (parseFloat(row[key]) / 100).toFixed(2) : parseFloat(row[key]);
                  });
              }
              return row;
          }).filter(Boolean);

          stats.cleanRows = processed.length;
          setHealthStats(stats);

          if (activeFormat.includes('python')) return generatePyTorchCode(activeExternal, processed);

          if (processed.length === 0) return "No data.";
          const keys = Object.keys(processed[0]);
          return [keys.join(','), ...processed.map((r: any) => keys.map(k => r[k]).join(','))].join('\n');
      }
      return "";
  };

  const generatePyTorchCode = (datasetName: string, sampleData: any[]) => {
      const isSequence = datasetName === 'progress';
      const isMultimodal = datasetName === 'emotion';
      
      return `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
${modelConfig.validationProtocol === 'k-fold' ? 'from sklearn.model_selection import KFold' : 'from sklearn.model_selection import train_test_split'}
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- AISLE AI LABORATORY CONFIG ---
# Task: ${modelConfig.taskType.toUpperCase()} | Dataset: ${datasetName.toUpperCase()}
BATCH_SIZE = ${modelConfig.batchSize}
EPOCHS = ${modelConfig.epochs}
LR = ${modelConfig.learningRate}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET CLASS ---
class AISLEDataset(Dataset):
    def __init__(self, data, target_col='score', is_sequence=${isSequence}):
        self.is_sequence = is_sequence
        # Feature Engineering based on AISLE Dataset Type
        if is_sequence:
            # Grouping by student for sequence modeling
            self.sequences = [g[1].drop(columns=[target_col]).values for g in data.groupby('student_id')]
            self.targets = data.groupby('student_id')[target_col].mean().values # Example target
        else:
            self.features = data.drop(columns=[target_col]).values
            self.targets = data[target_col].values

    def __len__(self):
        return len(self.sequences) if self.is_sequence else len(self.features)

    def __getitem__(self, idx):
        if self.is_sequence:
            return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# --- MODEL ARCHITECTURE (${modelConfig.algorithm.toUpperCase()}) ---
class AISLEModel(nn.Module):
    def __init__(self, input_dim):
        super(AISLEModel, self).__init__()
        
        ${modelConfig.algorithm === 'lstm' ? `
        # Recurrent Architecture for Learning Tracing
        self.lstm = nn.LSTM(input_dim, ${modelConfig.layerSize}, num_layers=${modelConfig.hiddenLayers}, batch_first=True, dropout=${modelConfig.dropout})
        self.fc = nn.Linear(${modelConfig.layerSize}, 1)
        ` : modelConfig.algorithm === 'cnn' ? `
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * ((input_dim - 2) // 2), 1)
        ` : `
        # Dense Architecture for Classification/Regression
        layers = []
        in_dim = input_dim
        for _ in range(${modelConfig.hiddenLayers}):
            layers.append(nn.Linear(in_dim, ${modelConfig.layerSize}))
            layers.append(nn.${modelConfig.activation === 'relu' ? 'ReLU()' : modelConfig.activation === 'tanh' ? 'Tanh()' : 'Sigmoid()'})
            layers.append(nn.Dropout(${modelConfig.dropout}))
            in_dim = ${modelConfig.layerSize}
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        `}

    def forward(self, x):
        ${modelConfig.algorithm === 'lstm' ? `
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last time step
        return self.fc(out)
        ` : modelConfig.algorithm === 'cnn' ? `
        x = x.unsqueeze(1) # Add channel dim
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
        ` : `
        return self.net(x)
        `}

# --- EXPERIMENTAL PROTOCOL ---
def run_experiment():
    print(f"Starting Experiment: {EPOCHS} Epochs, Optimizer: ${modelConfig.optimizer.toUpperCase()}")
    
    # Load & Preprocess
    df = pd.read_csv('aisle_${datasetName}.csv')
    # ... (Preprocessing simulation) ...
    
    # Protocol: ${modelConfig.validationProtocol.toUpperCase()}
    ${modelConfig.validationProtocol === 'k-fold' ? `
    kf = KFold(n_splits=${modelConfig.kFoldSplits}, shuffle=True)
    fold_results = []
    # K-Fold Loop logic...
    ` : `
    # Holdout Split logic...
    model = AISLEModel(input_dim=10).to(DEVICE)
    optimizer = optim.${modelConfig.optimizer === 'adam' ? 'Adam' : modelConfig.optimizer === 'sgd' ? 'SGD' : 'RMSprop'}(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        # Train batch...
        
        # Validation
        model.eval()
        # Eval batch...
    `}

if __name__ == "__main__":
    run_experiment()
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
            <BrainCircuit size={32} className="text-indigo-600" />
            AI Laboratory & Dataset Engine
            </h2>
            <p className="text-slate-600 mt-2">Design, preprocess, and generate AI model architectures for AISLE data.</p>
        </div>
        <div className="flex bg-slate-100 p-1 rounded-lg">
            <button onClick={() => setDataSource('graph')} className={`px-4 py-2 text-xs font-bold uppercase rounded-md transition-all ${dataSource === 'graph' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500'}`}>Graph</button>
            <button onClick={() => { setDataSource('external'); setActiveFormat('python_pytorch'); }} className={`px-4 py-2 text-xs font-bold uppercase rounded-md transition-all ${dataSource === 'external' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500'}`}>Model Lab</button>
        </div>
      </div>

      <div className="flex-1 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden flex flex-col md:flex-row">
          
          {/* Controls Sidebar */}
          <div className="w-full md:w-80 bg-slate-50 border-r border-slate-200 flex flex-col overflow-y-auto">
              
              {dataSource === 'external' && (
                  <div className="p-4 border-b border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-3 flex items-center gap-2"><Database size={12}/> Select Dataset</div>
                      <select className="w-full p-2 text-sm border rounded bg-white font-medium text-slate-700 focus:ring-2 focus:ring-indigo-500 outline-none" value={activeExternal} onChange={(e) => setActiveExternal(e.target.value as any)}>
                          <option value="braille">Braille Learners (Tabular)</option>
                          <option value="content">Course Content (Text)</option>
                          <option value="progress">Student Progress (Time-Series)</option>
                          <option value="emotion">Emotion Multimodal (Image/Text)</option>
                          <option value="students">Student Profiles (Tabular)</option>
                      </select>
                  </div>
              )}

              {/* AI MODEL LAB CONTROLS */}
              {activeFormat.includes('python') && (
                  <div className="flex-1 overflow-y-auto">
                      <div className="p-4 bg-indigo-50/50 border-b border-indigo-100">
                          <div className="text-xs font-bold text-indigo-700 uppercase mb-4 flex items-center gap-2"><Beaker size={14}/> Model Architecture</div>
                          
                          <div className="space-y-4">
                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Algorithm</label>
                                  <select className="w-full p-2 text-xs border rounded bg-white" value={modelConfig.algorithm} onChange={e => setModelConfig({...modelConfig, algorithm: e.target.value})}>
                                      {AI_ALGORITHMS.map(alg => <option key={alg.id} value={alg.id}>{alg.name}</option>)}
                                  </select>
                              </div>

                              <div className="grid grid-cols-2 gap-3">
                                  <div>
                                      <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Hidden Layers</label>
                                      <input type="number" min="1" max="10" value={modelConfig.hiddenLayers} onChange={e => setModelConfig({...modelConfig, hiddenLayers: parseInt(e.target.value)})} className="w-full p-1.5 text-xs border rounded"/>
                                  </div>
                                  <div>
                                      <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Units/Layer</label>
                                      <input type="number" step="16" value={modelConfig.layerSize} onChange={e => setModelConfig({...modelConfig, layerSize: parseInt(e.target.value)})} className="w-full p-1.5 text-xs border rounded"/>
                                  </div>
                              </div>

                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Activation Function</label>
                                  <div className="flex bg-white rounded border border-slate-200 p-1">
                                      {['relu', 'tanh', 'sigmoid'].map(act => (
                                          <button key={act} onClick={() => setModelConfig({...modelConfig, activation: act as any})} className={`flex-1 py-1 text-[9px] uppercase font-bold rounded ${modelConfig.activation === act ? 'bg-indigo-100 text-indigo-700' : 'text-slate-400 hover:text-slate-600'}`}>{act}</button>
                                      ))}
                                  </div>
                              </div>

                              <div>
                                  <div className="flex justify-between mb-1">
                                      <label className="text-[10px] font-bold text-slate-500 uppercase">Dropout Rate</label>
                                      <span className="text-[10px] text-slate-600 font-mono">{modelConfig.dropout}</span>
                                  </div>
                                  <input type="range" min="0" max="0.9" step="0.1" value={modelConfig.dropout} onChange={e => setModelConfig({...modelConfig, dropout: parseFloat(e.target.value)})} className="w-full h-1.5 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"/>
                              </div>
                          </div>
                      </div>

                      <div className="p-4 border-b border-slate-200">
                          <div className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2"><Settings2 size={14}/> Training Config</div>
                          
                          <div className="space-y-4">
                              <div className="grid grid-cols-2 gap-3">
                                  <div>
                                      <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Optimizer</label>
                                      <select className="w-full p-1.5 text-xs border rounded bg-white" value={modelConfig.optimizer} onChange={e => setModelConfig({...modelConfig, optimizer: e.target.value as any})}>
                                          <option value="adam">Adam</option>
                                          <option value="sgd">SGD</option>
                                          <option value="rmsprop">RMSprop</option>
                                      </select>
                                  </div>
                                  <div>
                                      <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Learning Rate</label>
                                      <input type="number" step="0.0001" value={modelConfig.learningRate} onChange={e => setModelConfig({...modelConfig, learningRate: parseFloat(e.target.value)})} className="w-full p-1.5 text-xs border rounded"/>
                                  </div>
                              </div>

                              <div>
                                  <label className="text-[10px] font-bold text-slate-500 uppercase block mb-1">Protocol</label>
                                  <div className="flex gap-2">
                                      <button onClick={() => setModelConfig({...modelConfig, validationProtocol: 'holdout'})} className={`flex-1 py-1.5 text-[10px] font-bold rounded border transition-colors ${modelConfig.validationProtocol === 'holdout' ? 'bg-slate-700 text-white border-slate-700' : 'bg-white text-slate-500 border-slate-200'}`}>Holdout</button>
                                      <button onClick={() => setModelConfig({...modelConfig, validationProtocol: 'k-fold'})} className={`flex-1 py-1.5 text-[10px] font-bold rounded border transition-colors ${modelConfig.validationProtocol === 'k-fold' ? 'bg-slate-700 text-white border-slate-700' : 'bg-white text-slate-500 border-slate-200'}`}>K-Fold CV</button>
                                  </div>
                              </div>
                          </div>
                      </div>
                  </div>
              )}
          </div>

          {/* Preview Panel */}
          <div className="flex-1 flex flex-col min-h-[500px]">
              <div className="p-3 border-b border-slate-100 flex justify-between items-center bg-white sticky top-0 z-10">
                  <div className="flex items-center gap-2 text-sm font-mono font-bold text-slate-600">
                      <Code2 size={16} className="text-emerald-500" /> 
                      {activeFormat === 'python_pytorch' ? 'model_training.py' : 'data_preview.csv'}
                  </div>
                  <div className="flex gap-2">
                      <select className="text-xs border rounded p-1" value={activeFormat} onChange={(e) => setActiveFormat(e.target.value as any)}>
                          <option value="python_pytorch">PyTorch Script</option>
                          <option value="csv">Clean CSV</option>
                      </select>
                      <button onClick={handleCopy} className="text-xs flex items-center gap-1 text-white bg-indigo-600 font-bold hover:bg-indigo-700 px-3 py-1 rounded transition-colors shadow-sm">
                          {copystate ? <Check size={12}/> : <Copy size={12} />} {copystate ? "Copied" : "Copy Code"}
                      </button>
                  </div>
              </div>
              <div className="flex-1 bg-slate-900 overflow-hidden relative p-6 overflow-y-auto custom-scrollbar">
                  <pre className={`font-mono text-xs whitespace-pre-wrap leading-relaxed ${activeFormat.includes('python') ? 'text-emerald-400' : 'text-slate-300'}`}>
                      {processedData}
                  </pre>
              </div>
          </div>
      </div>
    </div>
  );
};
