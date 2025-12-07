import React, { useState } from 'react';
import { analyzeFramework } from '../services/geminiService';
import { Sparkles, MessageSquare, BookOpen, AlertCircle, ArrowRight, Loader2, GitMerge, Award, Zap, CheckCircle, Users, Database } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export const AiAnalysis: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const predefinedPrompts = [
    {
        title: "Validate Conceptual Alignment",
        icon: <GitMerge size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "Validate whether each input aligns as: WCAG (the What), UDL (the Why), and CDP (the How). Interpret each node's functionality, its conceptual framework implication, and the cascading flow from intermediate processes to end product outcomes."
    },
    {
        title: "Academic Alignment Strategy",
        icon: <Award size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "How does the AISLE framework function as a 'Collaborative AI-Driven Learning Platform'? Explain how UDL, WCAG, and CDP collectively converge through intermediate processes (Efficacy, Resilience, Adoption) to enable maximum 'Academic Performance'."
    },
    {
        title: "Choreographical Validation",
        icon: <BookOpen size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "Validate the choreography of the 'I: Frame' approach. Check the 'Event' to 'State' transitions defined in the Orchestration logic for UDL, WCAG, and CDP. Ensure the dataset references (DS01-DS05) are correctly sequenced in the predecessor->descendant flow."
    },
    {
        title: "Validate Dataset Feature Bindings",
        icon: <Database size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "Validate the 'Key-Value Features' populated for each node. Check if the specific bindings (e.g., 'Audio Input' -> 'Librosa') correctly align with the assigned Datasets (DS01-DS05). Confirm that these technical features effectively enable the node's function within the AISLE framework."
    },
    {
        title: "Full Validation & Impact Report",
        icon: <CheckCircle size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "Perform a comprehensive validation step: 1) Trace forward/backward paths for logical consistency. 2) Evaluate the 'Green Computing' optimization and its necessity. 3) List the PROS and CONS of this architecture. 4) Assess the STRENGTH of the alignment to 'Academic Performance' - are all nodes effectively contributing? Interpret the final optimized graph."
    },
    {
        title: "Stakeholder Use Cases",
        icon: <Users size={20} className="text-slate-600 group-hover:text-indigo-600" />,
        query: "Based on the graph structure, suggest concrete Use Cases for the major actors: 1) VI Student (using Sensory/Emotion nodes), 2) Teacher (creating via Meta/UDL), 3) Developer (implementing Green/Privacy), and 4) Policymaker (evaluating CDP/Sustain). Explain how each actor interacts with their respective nodes to achieve the project goals."
    }
  ];

  const handlePrompt = async (query: string) => {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await analyzeFramework(query);
      setResponse(result);
    } catch (e) {
      setError("Failed to generate analysis. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center p-3 bg-indigo-50 rounded-full mb-4">
            <Sparkles className="text-indigo-600 w-8 h-8" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900 mb-2">Framework Intelligence</h2>
        <p className="text-slate-600 max-w-lg mx-auto">
          Use the AI assistant to critique, interpret, and validate the AISLE framework structure against its educational goals.
        </p>
      </div>

      {/* Suggested Actions */}
      {!response && !loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 mb-8">
            {predefinedPrompts.map((item, idx) => (
                <button 
                    key={idx}
                    onClick={() => handlePrompt(item.query)}
                    className="flex flex-col items-start p-6 bg-white border border-slate-200 rounded-xl hover:shadow-lg hover:border-indigo-300 transition-all text-left group"
                >
                    <div className="p-2 bg-slate-100 rounded-lg group-hover:bg-indigo-100 mb-4 transition-colors">
                        {item.icon}
                    </div>
                    <h3 className="font-semibold text-slate-900 mb-2">{item.title}</h3>
                    <p className="text-sm text-slate-500 mb-4 flex-1">{item.query}</p>
                    <div className="flex items-center text-xs font-bold text-indigo-600 mt-auto">
                        GENERATE REPORT <ArrowRight size={12} className="ml-1" />
                    </div>
                </button>
            ))}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="animate-spin text-indigo-600 w-10 h-10 mb-4" />
            <p className="text-slate-600 font-medium">Analyzing framework structure and relationships...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3 text-red-700 mb-6">
            <AlertCircle size={20} />
            {error}
        </div>
      )}

      {/* Results Display */}
      {response && (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
                <h3 className="font-bold text-slate-800 flex items-center gap-2">
                    <MessageSquare size={18} className="text-indigo-500"/> 
                    AI Analysis Report
                </h3>
                <button 
                    onClick={() => setResponse(null)}
                    className="text-xs font-medium text-slate-500 hover:text-slate-800"
                >
                    CLEAR REPORT
                </button>
            </div>
            <div className="p-8 prose prose-indigo max-w-none text-slate-700">
                <ReactMarkdown>{response}</ReactMarkdown>
            </div>
        </div>
      )}
    </div>
  );
};