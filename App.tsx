
import React, { useState, useEffect, useMemo } from 'react';
import { NetworkGraph } from './components/NetworkGraph';
import { NodeDetails } from './components/NodeDetails';
import { AiAnalysis } from './components/AiAnalysis';
import { DatasetBuilder } from './components/DatasetBuilder';
import { PlatformAccess } from './components/PlatformAccess';
import { AISLE_NODES, AISLE_LINKS } from './constants';
import { AisleNode, AisleLink } from './types';
import { suggestMissingLinks } from './services/geminiService';
import { Share2, Activity, BrainCircuit, Filter, Layers, Eye, EyeOff, Target, Scissors, Zap, CheckCircle2, Users, PlayCircle, BarChart3, Lightbulb, Database, Sparkles, Loader2, Fingerprint, Link } from 'lucide-react';

const App: React.FC = () => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'visualize' | 'analysis' | 'dataset' | 'platform'>('visualize');
  
  // Node Data State with Local Storage Persistence
  const [nodes, setNodes] = useState<AisleNode[]>(() => {
    try {
      // Updated to V19 to reflect enhanced data quality metrics
      const saved = localStorage.getItem('AISLE_NODES_V19');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.error("Failed to load nodes from local storage", e);
    }
    return AISLE_NODES;
  });

  const handleUpdateNode = (updatedNode: AisleNode) => {
    let newNodes = nodes.map(n => n.id === updatedNode.id ? updatedNode : n);
    
    if (simulationMode && updatedNode.simulation && updatedNode.simulation.value !== nodes.find(n=>n.id === updatedNode.id)?.simulation?.value) {
        newNodes = calculateSimulation(newNodes);
    }

    setNodes(newNodes);
    localStorage.setItem('AISLE_NODES_V19', JSON.stringify(newNodes));
  };

  const [isEditing, setIsEditing] = useState(false);
  const [visibleGroups, setVisibleGroups] = useState({ input: true, process: true, output: true, context: true });
  const [showResearchCodes, setShowResearchCodes] = useState(false);
  const [pruneWeakLinks, setPruneWeakLinks] = useState(false);
  const [connectDisconnected, setConnectDisconnected] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [aiSuggestions, setAiSuggestions] = useState<AisleLink[]>([]);
  const [isGeneratingLinks, setIsGeneratingLinks] = useState(false);
  const [activeAiLink, setActiveAiLink] = useState<AisleLink | null>(null);
  const [highlightTarget, setHighlightTarget] = useState<string | null>(null);
  const [selectedStakeholder, setSelectedStakeholder] = useState<string | null>(null);
  const [simulationMode, setSimulationMode] = useState(false);
  const [activeLinks, setActiveLinks] = useState<AisleLink[]>(AISLE_LINKS);

  const toggleGroup = (group: keyof typeof visibleGroups) => {
    setVisibleGroups(prev => ({ ...prev, [group]: !prev[group] }));
  };

  const handleGenerateAiLinks = async () => {
      setIsGeneratingLinks(true);
      try {
          const suggestions = await suggestMissingLinks(activeLinks);
          setAiSuggestions(suggestions);
      } catch (e) {
          console.error(e);
      } finally {
          setIsGeneratingLinks(false);
      }
  };

  const handleAcceptLink = (link: AisleLink) => {
      const newLink = { ...link, type: 'enables', isSuggested: false, isAiGenerated: false, label: 'e' }; 
      setActiveLinks(prev => [...prev, newLink]);
      setAiSuggestions(prev => prev.filter(l => !(l.source === link.source && l.target === link.target)));
      setActiveAiLink(null);
  };

  const handleRejectLink = (link: AisleLink) => {
      setAiSuggestions(prev => prev.filter(l => !(l.source === link.source && l.target === link.target)));
      setActiveAiLink(null);
  };

  const heuristicSuggestions = useMemo(() => {
    if (!showSuggestions) return [];
    const suggestions: AisleLink[] = [];
    const groupOrder: Record<string, number> = { input: 0, process: 1, output: 2, context: -1 };
    const existingLinks = new Set<string>();
    activeLinks.forEach(l => { existingLinks.add(`${l.source}-${l.target}`); existingLinks.add(`${l.target}-${l.source}`); });

    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const n1 = nodes[i];
            const n2 = nodes[j];
            if (existingLinks.has(`${n1.id}-${n2.id}`)) continue;
            let score = 0;
            const objs1 = n1.research?.objectives || [];
            const objs2 = n2.research?.objectives || [];
            if (objs1.filter(o => objs2.includes(o)).length > 0) score += objs1.filter(o => objs2.includes(o)).length * 2;
            const st1 = n1.stakeholders || [];
            const st2 = n2.stakeholders || [];
            if (st1.filter(s => st2.includes(s)).length > 0) score += st1.filter(s => st2.includes(s)).length * 0.5;

            if (score >= 3) {
                let source = n1.id;
                let target = n2.id;
                if (groupOrder[n1.group] !== groupOrder[n2.group] && groupOrder[n1.group] !== -1 && groupOrder[n2.group] !== -1) {
                     if (groupOrder[n1.group] > groupOrder[n2.group]) { source = n2.id; target = n1.id; }
                }
                suggestions.push({ source, target, type: 'suggested', label: '?', isSuggested: true, weight: 0.1 });
            }
        }
    }
    return suggestions;
  }, [nodes, showSuggestions, activeLinks]);

  const calculateSimulation = (currentNodes: AisleNode[]) => {
      const nextNodes = JSON.parse(JSON.stringify(currentNodes)) as AisleNode[];
      const nodeMap = new Map<string, AisleNode>();
      nextNodes.forEach(n => { if (!n.simulation) n.simulation = { value: 0 }; nodeMap.set(n.id, n); });

      for (let pass = 0; pass < 3; pass++) {
          nextNodes.forEach(targetNode => {
              if (targetNode.simulation?.isFixed) return;
              const incoming = activeLinks.filter(l => l.target === targetNode.id);
              if (incoming.length === 0) return;
              let weightedSum = 0;
              let totalWeight = 0;
              incoming.forEach(link => {
                  const sourceNode = nodeMap.get(link.source);
                  if (sourceNode && sourceNode.simulation) {
                      const weight = link.weight || 0.5;
                      weightedSum += sourceNode.simulation.value * weight;
                      totalWeight += weight;
                  }
              });
              if (totalWeight > 0) targetNode.simulation!.value = Math.min(100, Math.max(0, weightedSum / totalWeight));
          });
      }
      return nextNodes;
  };

  const toggleSimulationMode = () => {
      if (!simulationMode) {
          const simNodes = calculateSimulation(nodes);
          setNodes(simNodes);
          setSimulationMode(true);
      } else {
          setSimulationMode(false);
      }
  };

  const handleValidateAcademic = () => {
      const isActive = highlightTarget === 'Perform';
      if (isActive) { setHighlightTarget(null); setPruneWeakLinks(false); } 
      else { setHighlightTarget('Perform'); setPruneWeakLinks(true); setSelectedStakeholder(null); }
  };

  const selectedNode = selectedNodeId ? nodes.find(n => n.id === selectedNodeId) || null : null;
  const filteredNodes = nodes.filter(n => visibleGroups[n.group as keyof typeof visibleGroups]);
  const baseLinks = activeLinks.filter(l => filteredNodes.some(n => n.id === l.source) && filteredNodes.some(n => n.id === l.target));
  const finalLinks = [...baseLinks, ...heuristicSuggestions, ...aiSuggestions];

  return (
    <div className="flex flex-col h-screen w-full bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg text-white"><Share2 size={24} /></div>
          <div><h1 className="text-xl font-bold text-slate-900 leading-tight">AISLE Framework</h1><p className="text-xs text-slate-500 font-medium">Ultra-Compact Architecture for Visually Impaired Education</p></div>
        </div>
        <nav className="flex bg-slate-100 p-1 rounded-lg">
          <button onClick={() => setActiveTab('visualize')} className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'visualize' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}><div className="flex items-center gap-2"><Activity size={16} /><span>Interactive Graph</span></div></button>
          <button onClick={() => setActiveTab('analysis')} className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'analysis' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}><div className="flex items-center gap-2"><BrainCircuit size={16} /><span>AI Analysis</span></div></button>
          <button onClick={() => setActiveTab('dataset')} className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'dataset' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}><div className="flex items-center gap-2"><Database size={16} /><span>Dataset Engine</span></div></button>
          <button onClick={() => setActiveTab('platform')} className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'platform' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}><div className="flex items-center gap-2"><Fingerprint size={16} /><span>Platform Demo</span></div></button>
        </nav>
      </header>

      <main className="flex-1 flex overflow-hidden relative">
        {activeTab === 'visualize' && (
          <>
            <div className="absolute top-4 left-4 z-10 flex flex-col gap-4 w-64 max-h-[calc(100vh-100px)] overflow-y-auto">
                <div className="bg-white/95 backdrop-blur rounded-lg border border-slate-200 shadow-sm overflow-hidden flex-shrink-0">
                    <div className="px-4 py-3 border-b border-slate-100 bg-slate-50 flex items-center gap-2"><Filter size={14} className="text-slate-500" /><h3 className="font-semibold text-xs text-slate-700 uppercase tracking-wide">View Filters</h3></div>
                    <div className="p-3 space-y-2">{(['input', 'process', 'output', 'context'] as const).map(group => (<button key={group} onClick={() => toggleGroup(group)} className={`flex items-center justify-between w-full px-2 py-1.5 rounded text-xs font-medium transition-all ${visibleGroups[group] ? 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100' : 'bg-slate-50 text-slate-400 hover:bg-slate-100'}`}><span className="capitalize flex items-center gap-2"><Layers size={12} /> {group}</span>{visibleGroups[group] ? <Eye size={12} /> : <EyeOff size={12} />}</button>))}</div>
                    <div className="px-4 py-2 border-t border-slate-100 bg-slate-50 flex items-center gap-2"><PlayCircle size={14} className="text-rose-500" /><h3 className="font-semibold text-xs text-slate-700 uppercase tracking-wide">Predictive Modeling</h3></div>
                    <div className="p-3 pt-0"><button onClick={toggleSimulationMode} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${simulationMode ? 'bg-rose-600 text-white border-rose-600 shadow-md ring-2 ring-rose-200' : 'bg-white text-slate-600 border-slate-200 hover:border-rose-300 hover:text-rose-600'}`}><span className="flex items-center gap-2"><BarChart3 size={12} /> Test Simulation Mode</span><div className={`w-2 h-2 rounded-full ${simulationMode ? 'bg-white' : 'bg-slate-300'}`}></div></button></div>
                    <div className="px-4 py-2 border-t border-slate-100 bg-slate-50 flex items-center gap-2"><Users size={14} className="text-indigo-500" /><h3 className="font-semibold text-xs text-slate-700 uppercase tracking-wide">Stakeholder View</h3></div>
                    <div className="p-3 pt-0 grid grid-cols-2 gap-2">{(['Student', 'Teacher', 'Developer', 'Policymaker'] as const).map(role => (<button key={role} onClick={() => { setSelectedStakeholder(prev => prev === role ? null : role); setHighlightTarget(null); }} className={`px-2 py-2 rounded text-[10px] font-bold uppercase tracking-wide border transition-all ${selectedStakeholder === role ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm' : 'bg-white text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600'}`}>{role}</button>))}</div>
                    <div className="px-4 py-2 border-t border-slate-100 bg-slate-50 flex items-center gap-2"><CheckCircle2 size={14} className="text-emerald-500" /><h3 className="font-semibold text-xs text-slate-700 uppercase tracking-wide">Validation</h3></div>
                    <div className="p-3 pt-0 space-y-2"><button onClick={handleValidateAcademic} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${highlightTarget === 'Perform' ? 'bg-emerald-50 text-emerald-700 border-emerald-200 ring-1 ring-emerald-200' : 'bg-white text-slate-600 border-slate-200 hover:border-emerald-300'}`}><span>Validate Academic Path</span>{highlightTarget === 'Perform' && <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>}</button></div>
                    <div className="px-4 py-2 border-t border-slate-100 bg-slate-50 flex items-center gap-2"><Scissors size={14} className="text-amber-500" /><h3 className="font-semibold text-xs text-slate-700 uppercase tracking-wide">Topology</h3></div>
                    <div className="p-3 pt-0 space-y-2"><button onClick={() => setPruneWeakLinks(!pruneWeakLinks)} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${pruneWeakLinks ? 'bg-amber-50 text-amber-700 border-amber-200' : 'bg-white text-slate-600 border-slate-200 hover:border-amber-300'}`}><span>Distill (Prune Weak)</span>{pruneWeakLinks ? <EyeOff size={12} /> : <Eye size={12} />}</button><button onClick={() => setConnectDisconnected(!connectDisconnected)} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${connectDisconnected ? 'bg-orange-50 text-orange-700 border-orange-200' : 'bg-white text-slate-600 border-slate-200 hover:border-orange-300'}`}><span className="flex items-center gap-1"><Link size={12} /> Connect Disconnected</span>{connectDisconnected && <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse"></div>}</button><button onClick={() => setShowSuggestions(!showSuggestions)} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${showSuggestions ? 'bg-yellow-50 text-yellow-700 border-yellow-200' : 'bg-white text-slate-600 border-slate-200 hover:border-yellow-300'}`}><span className="flex items-center gap-1"><Lightbulb size={12} /> Heuristic Suggest</span>{showSuggestions && <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>}</button><button onClick={handleGenerateAiLinks} disabled={isGeneratingLinks} className={`flex items-center justify-between w-full px-2 py-2 rounded text-xs font-medium border transition-all ${aiSuggestions.length > 0 ? 'bg-violet-50 text-violet-700 border-violet-200' : 'bg-white text-slate-600 border-slate-200 hover:border-violet-300'}`}><span className="flex items-center gap-1">{isGeneratingLinks ? <Loader2 size={12} className="animate-spin"/> : <Sparkles size={12} />} Generate AI Links</span>{aiSuggestions.length > 0 && <span className="text-[10px] bg-violet-200 text-violet-800 px-1 rounded-full">{aiSuggestions.length}</span>}</button></div>
                    <div className="px-4 py-3 border-t border-slate-100 bg-slate-50 mt-auto"><div className="flex items-center justify-between"><span className="text-xs font-medium text-slate-600">Show Research Codes</span><button onClick={() => setShowResearchCodes(!showResearchCodes)} className={`w-8 h-4 rounded-full transition-colors relative ${showResearchCodes ? 'bg-indigo-600' : 'bg-slate-300'}`}><div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${showResearchCodes ? 'translate-x-4' : ''}`}></div></button></div></div>
                    <div className="p-4 border-t border-slate-200 bg-white"><h3 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2">Node Legend</h3><div className="grid grid-cols-2 gap-2 text-[10px]"><div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-blue-100 border border-blue-500"></div><span>Input (Blue)</span></div><div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-green-100 border border-green-500"></div><span>Process (Green)</span></div><div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-red-100 border border-red-500"></div><span>Output (Red)</span></div><div className="flex items-center gap-1"><div className="w-2 h-2 rounded bg-slate-100 border border-slate-500"></div><span>Context (Gray)</span></div></div><h3 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2 mt-3">Link Legend</h3><div className="space-y-1 text-[10px]"><div className="flex items-center gap-2"><div className="w-6 h-0.5 bg-blue-500"></div><span>Guides (Input)</span></div><div className="flex items-center gap-2"><div className="w-6 h-0.5 bg-emerald-500"></div><span>Enables/Creates</span></div><div className="flex items-center gap-2"><div className="w-6 h-0.5 bg-rose-500 border-b border-dashed border-rose-500"></div><span>Predicts (Outcome)</span></div><div className="flex items-center gap-2"><div className="w-6 h-0.5 border-t-2 border-dotted border-slate-400"></div><span>Supports/Context</span></div></div></div>
                </div>
            </div>
            <div className="flex-1 h-full bg-slate-50 relative" onClick={() => { setSelectedNodeId(null); setIsEditing(false); setActiveAiLink(null); }}>
                <NetworkGraph 
                    nodes={filteredNodes} 
                    links={finalLinks} 
                    onNodeClick={(node) => { setSelectedNodeId(node.id); if (!visibleGroups[node.group as keyof typeof visibleGroups]) setVisibleGroups(prev => ({ ...prev, [node.group]: true })); }} 
                    onNodeDoubleClick={(node) => { setSelectedNodeId(node.id); setIsEditing(true); }} 
                    onLinkClick={(link) => { if (link.isAiGenerated) setActiveAiLink(link); }} 
                    selectedNodeId={selectedNodeId} 
                    highlightTargetId={highlightTarget} 
                    showResearchCodes={showResearchCodes} 
                    pruneWeakLinks={pruneWeakLinks} 
                    connectDisconnected={connectDisconnected} 
                    selectedStakeholder={selectedStakeholder} 
                    simulationMode={simulationMode}
                    activeAiLink={activeAiLink}
                />
                {activeAiLink && (<div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-4 rounded-xl shadow-2xl border border-violet-200 z-30 max-w-sm animate-in fade-in zoom-in-95"><div className="flex items-center gap-2 mb-3 text-violet-700 font-bold uppercase text-xs tracking-wider"><Sparkles size={14} /> AI Suggested Connection</div><div className="flex items-center gap-2 justify-center mb-4 text-sm font-semibold text-slate-800"><span className="bg-slate-100 px-2 py-1 rounded">{activeAiLink.source}</span><span className="text-slate-400">→</span><span className="bg-slate-100 px-2 py-1 rounded">{activeAiLink.target}</span></div><p className="text-xs text-slate-600 mb-4 bg-slate-50 p-3 rounded italic border border-slate-100">"{activeAiLink.reason}"</p><div className="flex gap-2"><button onClick={() => handleRejectLink(activeAiLink)} className="flex-1 py-2 text-xs font-bold text-slate-500 bg-slate-100 hover:bg-slate-200 rounded">Reject</button><button onClick={() => handleAcceptLink(activeAiLink)} className="flex-1 py-2 text-xs font-bold text-white bg-violet-600 hover:bg-violet-700 rounded shadow-sm">Accept Connection</button></div></div>)}
            </div>
          </>
        )}
        {activeTab === 'analysis' && <div className="w-full h-full overflow-y-auto bg-slate-50/50"><AiAnalysis /></div>}
        {activeTab === 'dataset' && <div className="w-full h-full overflow-y-auto bg-slate-50/50"><DatasetBuilder nodes={nodes} links={AISLE_LINKS} /></div>}
        {activeTab === 'platform' && <div className="w-full h-full overflow-y-auto bg-slate-50/50"><PlatformAccess /></div>}
        {selectedNode && activeTab === 'visualize' && (<div className={`absolute top-4 right-4 bottom-4 w-[450px] bg-white rounded-xl shadow-2xl border border-slate-200 transform transition-transform duration-300 z-20 flex flex-col ${selectedNode ? 'translate-x-0' : 'translate-x-full'}`}><NodeDetails node={selectedNode} onClose={() => { setSelectedNodeId(null); setIsEditing(false); }} onUpdateNode={handleUpdateNode} isEditing={isEditing} onToggleEdit={setIsEditing} simulationMode={simulationMode} /></div>)}
      </main>
    </div>
  );
};

export default App;
