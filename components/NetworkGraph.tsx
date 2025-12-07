
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { AisleNode, AisleLink } from '../types';
import { Sparkles, Loader2, Check, X, Wand2 } from 'lucide-react';

interface NetworkGraphProps {
  nodes: AisleNode[];
  links: AisleLink[];
  onNodeClick: (node: AisleNode) => void;
  onNodeDoubleClick?: (node: AisleNode) => void;
  onLinkClick?: (link: AisleLink) => void;
  selectedNodeId: string | null;
  highlightTargetId: string | null;
  showResearchCodes: boolean;
  pruneWeakLinks?: boolean;
  connectDisconnected?: boolean;
  selectedStakeholder?: string | null;
  simulationMode?: boolean;
  activeAiLink?: AisleLink | null;
  // AI Feature Props
  onGenerateAiLinks?: () => void;
  onAcceptAiLink?: (link: AisleLink) => void;
  onRejectAiLink?: (link: AisleLink) => void;
  isGeneratingAiLinks?: boolean;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({ 
    nodes, 
    links, 
    onNodeClick, 
    onNodeDoubleClick, 
    onLinkClick, 
    selectedNodeId, 
    highlightTargetId, 
    showResearchCodes, 
    pruneWeakLinks = false, 
    connectDisconnected = false, 
    selectedStakeholder, 
    simulationMode,
    activeAiLink,
    onGenerateAiLinks,
    onAcceptAiLink,
    onRejectAiLink,
    isGeneratingAiLinks
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Helper to safely get ID from link source/target
  const getId = (n: any): string => {
      if (!n) return "";
      return (typeof n === 'object' ? n.id : n) as string;
  };

  // Helper to normalize a link object for callbacks (ensures source/target are strings)
  const normalizeLink = (l: any): AisleLink => {
      return {
          ...l,
          source: getId(l.source),
          target: getId(l.target)
      };
  };

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    
    // Create a mutable copy of the data for d3 to modify
    const graphNodes = nodes.map(d => ({ ...d }));
    
    // Create a lookup map for faster access to node data
    const nodeMap = new Map<string, AisleNode>();
    graphNodes.forEach(n => nodeMap.set(n.id, n));

    // Helper to safely compare links (handling D3 object vs String ID references)
    const getLinkId = (l: any) => {
        if (!l) return "";
        const s = getId(l.source);
        const t = getId(l.target);
        return `${s}-${t}`;
    };

    const activeAiLinkId = activeAiLink ? getLinkId(activeAiLink) : null;

    // Helper: Check if a node is relevant to the selected stakeholder
    const isRelevantNode = (nodeId: string | AisleNode) => {
        if (!selectedStakeholder) return true;
        const id = (typeof nodeId === 'object' && 'id' in nodeId) ? nodeId.id : nodeId as string;
        const n = nodeMap.get(id);
        return (n?.stakeholders || []).includes(selectedStakeholder);
    };

    // Helper: Check if a link is relevant to the selected stakeholder
    const isRelevantLink = (l: AisleLink | any) => {
        if (!selectedStakeholder) return true;
        const sId = getId(l.source);
        const tId = getId(l.target);
        return isRelevantNode(sId) && isRelevantNode(tId);
    };
    
    // Filter Links based on "Pruning" logic
    let graphLinks = links.filter(l => {
        if (!pruneWeakLinks) return true;
        const weakTypes = ['supports', 'contextualizes', 'informs'];
        return !weakTypes.includes(l.type);
    }).map(d => ({ ...d }));

    // --- GRAPH HEALING LOGIC ---
    if (connectDisconnected) {
        const connectedNodeIds = new Set<string>();
        graphLinks.forEach(l => {
            connectedNodeIds.add(getId(l.source));
            connectedNodeIds.add(getId(l.target));
        });

        const hubId = 'Green'; 
        const isolatedNodes = graphNodes.filter(n => !connectedNodeIds.has(n.id) && n.id !== hubId);
        const hubNode = graphNodes.find(n => n.id === hubId);

        if (hubNode && isolatedNodes.length > 0) {
            const healingLinks = isolatedNodes.map(n => ({
                source: hubId,
                target: n.id,
                type: 'healing',
                label: 'heal',
                isOptimization: true,
                isSuggested: true
            }));
            graphLinks = [...graphLinks, ...healingLinks] as any[];
        }
    }

    const groups = {
      input: width * 0.15,
      process: width * 0.5,
      output: width * 0.85,
      context: width * 0.5
    };

    // --- HIGHLIGHTING LOGIC ---
    const traceHighlightedIds = new Set<string>();
    const traceHighlightedLinkIndices = new Set<number>();
    
    if (highlightTargetId) {
      traceHighlightedIds.add(highlightTargetId);
      const queue = [highlightTargetId];
      const visited = new Set<string>([highlightTargetId]);
      const isInput = nodes.find(n => n.id === highlightTargetId)?.group === 'input';
      
      if (isInput) {
           while(queue.length > 0) {
               const currentId = queue.shift()!;
               const outgoingLinks = graphLinks.filter(l => getId(l.source) === currentId);
               outgoingLinks.forEach(l => {
                   const tId = getId(l.target);
                   if (!visited.has(tId)) {
                       visited.add(tId);
                       traceHighlightedIds.add(tId);
                       queue.push(tId);
                   }
               });
           }
            graphLinks.forEach((l, idx) => {
                const sId = getId(l.source);
                const tId = getId(l.target);
                if (traceHighlightedIds.has(sId) && traceHighlightedIds.has(tId)) {
                    traceHighlightedLinkIndices.add(idx);
                }
            });
      } else {
          while (queue.length > 0) {
            const currentId = queue.shift()!;
            const incomingLinks = graphLinks.filter(l => getId(l.target) === currentId);
            incomingLinks.forEach(l => {
              const sId = getId(l.source);
              if (!visited.has(sId)) {
                visited.add(sId);
                traceHighlightedIds.add(sId);
                queue.push(sId);
              }
            });
          }
          graphLinks.forEach((l, idx) => {
             const sId = getId(l.source);
             const tId = getId(l.target);
             if (traceHighlightedIds.has(sId) && traceHighlightedIds.has(tId)) {
                 traceHighlightedLinkIndices.add(idx);
             }
          });
      }
    }

    const selectedNeighborIds = new Set<string>();
    const selectedNeighborLinkIndices = new Set<number>();

    if (selectedNodeId) {
        selectedNeighborIds.add(selectedNodeId);
        graphLinks.forEach((l, i) => {
            const sId = getId(l.source);
            const tId = getId(l.target);
            if (sId === selectedNodeId || tId === selectedNodeId) {
                selectedNeighborLinkIndices.add(i);
                selectedNeighborIds.add(sId);
                selectedNeighborIds.add(tId);
            }
        });
    }

    const svg = d3.select(svgRef.current)
      .attr("viewBox", [0, 0, width, height])
      .style("max-width", "100%")
      .style("height", "auto");

    const typeConfig: Record<string, { color: string; dash: string; width: number; animate?: boolean }> = {
      guides: { color: "#3b82f6", dash: "none", width: 2, animate: false },
      enables: { color: "#10b981", dash: "none", width: 2.5, animate: false },
      creates: { color: "#10b981", dash: "none", width: 2.5, animate: false },
      predicts: { color: "#f43f5e", dash: "6,3", width: 2, animate: true },
      supports: { color: "#64748b", dash: "3,3", width: 1.5, animate: false },
      contextualizes: { color: "#94a3b8", dash: "4,4", width: 1.5, animate: false },
      informs: { color: "#8b5cf6", dash: "5,5", width: 1.5, animate: true },
      suggested: { color: "#eab308", dash: "4,2", width: 2, animate: true },
      healing: { color: "#f59e0b", dash: "2,2", width: 2, animate: true },
      default: { color: "#94a3b8", dash: "none", width: 1.5, animate: false }
    };

    const getTypeStyle = (d: any) => {
        if (d.isAiGenerated) return { color: "#8b5cf6", dash: "4,2", width: 2.5, animate: true };
        return typeConfig[d.type] || typeConfig.default;
    };

    const defs = svg.append("defs");
    Object.keys(typeConfig).forEach(type => {
      const config = typeConfig[type];
      defs.append("marker")
        .attr("id", `arrow-${type}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 28)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("fill", config.color)
        .attr("d", "M0,-5L10,0L0,5");

      defs.append("marker")
        .attr("id", `arrow-${type}-dimmed`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 28)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("fill", "#e2e8f0")
        .attr("d", "M0,-5L10,0L0,5");
    });

    defs.append("marker")
        .attr("id", "arrow-ai")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 28)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("fill", "#8b5cf6")
        .attr("d", "M0,-5L10,0L0,5");

    const simulation = d3.forceSimulation(graphNodes as any)
      .force("link", d3.forceLink(graphLinks).id((d: any) => d.id).distance(180)) 
      .force("charge", d3.forceManyBody().strength(-1200)) 
      .force("collide", d3.forceCollide().radius(80)) 
      .force("x", d3.forceX((d: any) => {
         if (d.group === 'input') return groups.input;
         if (d.group === 'process') return groups.process;
         if (d.group === 'output') return groups.output;
         return groups.context;
      }).strength(0.8))
      .force("y", d3.forceY((d: any) => {
          if (d.group === 'context') return height * 0.9;
          return height / 2;
      }).strength(0.2));

    const link = svg.append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(graphLinks)
      .join("path")
      .attr("class", (d: any) => {
         const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
         const style = getTypeStyle(d);
         if (isActiveAi) return "pulse-glow";
         return style.animate ? "animate-dash-flow" : "";
      })
      .attr("stroke-width", (d: any, i) => {
          const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
          if (isActiveAi) return 4;
          
          const style = getTypeStyle(d);
          if (d.isAiGenerated) return 4;
          if (selectedNodeId && selectedNeighborLinkIndices.has(i)) return style.width + 1.5;
          if (highlightTargetId && traceHighlightedLinkIndices.has(i)) return style.width + 1;
          return style.width;
      })
      .attr("stroke", (d: any, i) => {
          const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
          if (isActiveAi) return "#7c3aed"; // Vivid Violet for active AI link

          const style = getTypeStyle(d);
          if (selectedNodeId) return selectedNeighborLinkIndices.has(i) ? style.color : "#e2e8f0";
          if (highlightTargetId) return traceHighlightedLinkIndices.has(i) ? style.color : "#e2e8f0";
          if (selectedStakeholder) return isRelevantLink(d) ? style.color : "#f1f5f9";
          return style.color;
      })
      .attr("opacity", (d: any, i) => {
          const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
          if (isActiveAi) return 1;

          if (selectedNodeId) return selectedNeighborLinkIndices.has(i) ? 1 : 0.05; // Stronger dimming
          if (highlightTargetId) return traceHighlightedLinkIndices.has(i) ? 1 : 0.1;
          if (selectedStakeholder) return isRelevantLink(d) ? 0.9 : 0.05;
          return 0.8;
      })
      .attr("cursor", (d: any) => d.isAiGenerated ? "pointer" : "default")
      .attr("stroke-dasharray", (d: any) => getTypeStyle(d).dash)
      .attr("marker-end", (d: any, i) => {
          if (d.isAiGenerated) return "url(#arrow-ai)";
          const type = d.type;
          const safeType = typeConfig[type] ? type : 'default';
          let isActive = true;
          if (selectedNodeId && !selectedNeighborLinkIndices.has(i)) isActive = false;
          else if (highlightTargetId && !traceHighlightedLinkIndices.has(i)) isActive = false;
          else if (selectedStakeholder && !isRelevantLink(d)) isActive = false;
          return isActive ? `url(#arrow-${safeType})` : `url(#arrow-${safeType}-dimmed)`;
      })
      .on("click", (event, d) => {
          if (d.isAiGenerated && onLinkClick) {
              event.stopPropagation();
              onLinkClick(normalizeLink(d));
          }
      });

    const linkLabelGroup = svg.append("g")
        .selectAll("g")
        .data(graphLinks)
        .join("g")
        .attr("opacity", (d: any, i) => {
            const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
            if (isActiveAi) return 1;
            if (selectedNodeId) return selectedNeighborLinkIndices.has(i) ? 1 : 0.05;
            if (highlightTargetId) return traceHighlightedLinkIndices.has(i) ? 1 : 0.1;
            if (selectedStakeholder) return isRelevantLink(d) ? 1 : 0.05;
            return 1;
        })
        .style("cursor", (d: any) => d.isAiGenerated ? "pointer" : "default")
        .on("click", (event, d) => {
            if (d.isAiGenerated && onLinkClick) {
                event.stopPropagation();
                onLinkClick(normalizeLink(d));
            }
        });

    linkLabelGroup.append("circle")
        .attr("r", 9)
        .attr("fill", (d: any, i) => {
             const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
             if (isActiveAi) return "#7c3aed"; 

             const style = getTypeStyle(d);
             if (selectedNodeId && selectedNeighborLinkIndices.has(i)) return style.color;
             if (highlightTargetId && traceHighlightedLinkIndices.has(i)) return style.color;
             if (selectedStakeholder && isRelevantLink(d)) return style.color;
             return (selectedNodeId || highlightTargetId || selectedStakeholder) ? "#f1f5f9" : style.color;
        })
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5);

    linkLabelGroup.append("text")
        .attr("font-size", "9px")
        .attr("fill", (d: any, i) => {
             const isActiveAi = activeAiLinkId && getLinkId(d) === activeAiLinkId;
             if (isActiveAi) return "#fff";

             const isActive = 
                (selectedNodeId && selectedNeighborLinkIndices.has(i)) ||
                (highlightTargetId && traceHighlightedLinkIndices.has(i)) ||
                (selectedStakeholder && isRelevantLink(d)) ||
                (!selectedNodeId && !highlightTargetId && !selectedStakeholder);
             return isActive ? "#fff" : "#cbd5e1";
        })
        .attr("font-weight", "bold")
        .attr("text-anchor", "middle")
        .attr("dy", 3)
        .text((d: any) => d.label || "");

    const tooltip = d3.select(tooltipRef.current);

    const node = svg.append("g")
      .selectAll("g")
      .data(graphNodes)
      .join("g")
      .attr("opacity", d => {
          if (selectedNodeId) return selectedNeighborIds.has(d.id) ? 1 : 0.1;
          if (highlightTargetId) return traceHighlightedIds.has(d.id) ? 1 : 0.1;
          if (selectedStakeholder) return isRelevantNode(d) ? 1 : 0.1;
          return 1;
      })
      .call(d3.drag<SVGGElement, any>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("mouseenter", function(event, d) {
          tooltip.style("display", "block").style("opacity", 1)
            .html(`
              <div class="font-semibold text-sm mb-1 border-b border-slate-700 pb-1">${d.fullLabel || d.label}</div>
              <div class="text-xs text-slate-300 leading-relaxed">${d.description}</div>
              ${d.conceptualRole ? `<div class="mt-2 text-[10px] uppercase font-bold text-indigo-300 bg-indigo-900/50 inline-block px-1 rounded">${d.conceptualRole}</div>` : ''}
              ${simulationMode && d.simulation ? `
                  <div class="mt-2 pt-1 border-t border-slate-700">
                    <div class="text-[10px] font-bold text-amber-400 uppercase">Simulated Value</div>
                    <div class="flex items-center gap-2">
                        <div class="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r from-amber-500 to-yellow-400" style="width: ${d.simulation.value}%"></div>
                        </div>
                        <span class="text-xs font-mono">${Math.round(d.simulation.value)}%</span>
                    </div>
                  </div>
              ` : ''}
              ${showResearchCodes && (d.research?.questions || d.research?.objectives) ? `
                <div class="mt-2 flex flex-wrap gap-1">
                  ${(d.research.objectives || []).map((o: string) => `<span class="px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[9px] font-mono border border-amber-500/50">${o}</span>`).join('')}
                  ${(d.research.questions || []).map((q: string) => `<span class="px-1 py-0.5 rounded bg-sky-500/20 text-sky-300 text-[9px] font-mono border border-sky-500/50">${q}</span>`).join('')}
                </div>
              ` : ''}
              ${d.stakeholders ? `
                <div class="mt-2 flex flex-wrap gap-1 border-t border-slate-700 pt-1">
                    ${d.stakeholders.map((s: string) => `<span class="text-[9px] text-slate-400 italic ${selectedStakeholder === s ? 'text-indigo-300 font-bold' : ''}">${s}</span>`).join('<span class="text-slate-600">•</span>')}
                </div>
              ` : ''}
            `);
            d3.select(this).select(".edit-button").transition().duration(200).attr("opacity", 1).attr("pointer-events", "all");
      })
      .on("mousemove", (event) => {
          const containerRect = containerRef.current?.getBoundingClientRect();
          if (containerRect) {
              const tooltipEl = tooltipRef.current;
              let top = event.clientY + 15;
              let left = event.clientX + 15;
              if (tooltipEl) {
                  const tooltipRect = tooltipEl.getBoundingClientRect();
                  if (left + tooltipRect.width > window.innerWidth - 20) left = event.clientX - tooltipRect.width - 10;
                  if (top + tooltipRect.height > window.innerHeight - 20) top = event.clientY - tooltipRect.height - 10;
              }
              tooltip.style("left", left + "px").style("top", top + "px");
          }
      })
      .on("mouseleave", function() {
          tooltip.style("opacity", 0).style("display", "none");
          d3.select(this).select(".edit-button").transition().duration(200).attr("opacity", 0).attr("pointer-events", "none");
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        onNodeClick(nodes.find(n => n.id === d.id)!);
      })
      .on("dblclick", (event, d) => {
         event.stopPropagation();
         if (onNodeDoubleClick) onNodeDoubleClick(nodes.find(n => n.id === d.id)!);
      });

    node.append("rect")
      .attr("width", d => d.id === selectedNodeId ? 105 : 90)
      .attr("height", d => d.id === selectedNodeId ? 82 : 70)
      .attr("x", d => d.id === selectedNodeId ? -52.5 : -45)
      .attr("y", d => d.id === selectedNodeId ? -41 : -35)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("fill", d => {
        if (d.group === 'input') return "#eff6ff"; 
        if (d.group === 'process') return "#f0fdf4"; 
        if (d.group === 'output') return "#fef2f2"; 
        return "#f1f5f9"; 
      })
      .attr("stroke", d => {
        if (d.id === selectedNodeId) return "#6366f1"; 
        if (highlightTargetId && traceHighlightedIds.has(d.id)) return "#6366f1";
        if (selectedStakeholder && isRelevantNode(d)) return "#6366f1"; 
        if (d.group === 'input') return "#3b82f6"; 
        if (d.group === 'process') return "#22c55e"; 
        if (d.group === 'output') return "#ef4444"; 
        return "#94a3b8"; 
      })
      .attr("stroke-width", d => {
          if (d.id === selectedNodeId) return 3.5;
          if (selectedNeighborIds.has(d.id)) return 2.5; 
          if (highlightTargetId && traceHighlightedIds.has(d.id)) return 2.5;
          if (selectedStakeholder && isRelevantNode(d)) return 3; 
          return 1.5;
      })
      .attr("class", "cursor-pointer transition-all duration-300 ease-in-out");
    
    const editBtnGroup = node.append("g")
        .attr("class", "edit-button")
        .attr("opacity", 0)
        .attr("pointer-events", "none") 
        .attr("transform", d => d.id === selectedNodeId ? "translate(24, -48)" : "translate(22, -45)")
        .attr("cursor", "pointer")
        .on("click", (event, d) => {
            event.stopPropagation();
            if (onNodeDoubleClick) onNodeDoubleClick(nodes.find(n => n.id === d.id)!);
        });

    editBtnGroup.append("rect").attr("width", 26).attr("height", 14).attr("rx", 3).attr("fill", "#4f46e5").attr("stroke", "#fff").attr("stroke-width", 1);
    editBtnGroup.append("text").attr("x", 13).attr("y", 10).attr("text-anchor", "middle").attr("font-size", "8px").attr("font-weight", "bold").attr("fill", "white").text("EDIT");

    const labelGroup = node.append("g").attr("opacity", d => (selectedStakeholder && !isRelevantNode(d)) ? 0 : 1);
    labelGroup.append("text").attr("x", 0).attr("y", -12).attr("text-anchor", "middle").attr("font-size", d => d.id === selectedNodeId ? "12px" : "11px").attr("font-weight", "bold").attr("fill", "#1e293b").attr("pointer-events", "none").text(d => d.label);
    labelGroup.append("text").attr("x", 0).attr("y", 2).attr("text-anchor", "middle").attr("font-size", "9px").attr("fill", "#6366f1").attr("font-weight", "500").attr("pointer-events", "none").text(d => d.fullLabel || "");
    labelGroup.append("text").attr("x", 0).attr("y", 14).attr("text-anchor", "middle").attr("font-size", "9px").attr("fill", "#64748b").attr("pointer-events", "none").text(d => d.category || "");

    node.each(function(d) {
        const el = d3.select(this);
        const isSelected = d.id === selectedNodeId;
        const hasResearch = (showResearchCodes || isSelected) && (d.research?.objectives?.length || d.research?.questions?.length);
        el.selectAll(".sim-badge").remove();
        if (selectedStakeholder && !isRelevantNode(d)) return;

        if (simulationMode && d.simulation) {
             const val = Math.round(d.simulation.value);
             const color = d3.interpolateRdYlGn(val / 100);
             el.append("rect").attr("class", "sim-badge").attr("x", isSelected ? 32 : 28).attr("y", isSelected ? -40 : -35).attr("width", 22).attr("height", 22).attr("rx", 11).attr("fill", color).attr("stroke", "#fff").attr("stroke-width", 2);
             el.append("text").attr("class", "sim-badge").attr("x", isSelected ? 43 : 39).attr("y", isSelected ? -25 : -20).attr("text-anchor", "middle").attr("font-size", "9px").attr("font-weight", "bold").attr("fill", val > 60 || val < 30 ? "#fff" : "#000").attr("dominant-baseline", "middle").text(val);
        } else if (hasResearch) {
             const code = d.research?.objectives?.[0] || d.research?.questions?.[0];
             el.append("rect").attr("x", -12).attr("y", isSelected ? 22 : 18).attr("width", 24).attr("height", 12).attr("rx", 3).attr("fill", "#4f46e5");
             el.append("text").attr("x", 0).attr("y", isSelected ? 31 : 27).attr("text-anchor", "middle").attr("font-size", "9px").attr("font-weight", "bold").attr("fill", "#ffffff").text(code || "");
        } else if (d.conceptualRole) {
            el.append("text").attr("x", 0).attr("y", isSelected ? 30 : 26).attr("text-anchor", "middle").attr("font-size", "8px").attr("font-weight", "bold").attr("fill", "#0f172a").attr("class", "uppercase tracking-tighter").text(d.conceptualRole);
        }
    });

    simulation.on("tick", () => {
      link.attr("d", (d: any) => `M${d.source.x},${d.source.y}C${d.source.x + 60},${d.source.y} ${d.target.x - 60},${d.target.y} ${d.target.x},${d.target.y}`);
      linkLabelGroup.attr("transform", (d: any) => `translate(${(d.source.x + d.target.x) / 2},${(d.source.y + d.target.y) / 2})`);
      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }
    function dragged(event: any, d: any) { d.fx = event.x; d.fy = event.y; }
    function dragended(event: any, d: any) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

    return () => { simulation.stop(); };
  }, [nodes, links, selectedNodeId, highlightTargetId, showResearchCodes, pruneWeakLinks, connectDisconnected, selectedStakeholder, simulationMode, activeAiLink]);

  return (
    <div ref={containerRef} className="w-full h-full bg-slate-50 relative group overflow-hidden">
        <style>
          {`
            @keyframes dash-flow {
              from { stroke-dashoffset: 20; }
              to { stroke-dashoffset: 0; }
            }
            @keyframes pulse-glow {
              0% { stroke-opacity: 1; stroke-width: 4px; filter: drop-shadow(0 0 2px #8b5cf6); }
              50% { stroke-opacity: 0.6; stroke-width: 6px; filter: drop-shadow(0 0 8px #8b5cf6); }
              100% { stroke-opacity: 1; stroke-width: 4px; filter: drop-shadow(0 0 2px #8b5cf6); }
            }
            .animate-dash-flow { animation: dash-flow 1s linear infinite; }
            .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
          `}
        </style>
        <svg ref={svgRef} className="w-full h-full block" />
        
        {/* UI Overlay: AI Link Acceptance */}
        {activeAiLink && onAcceptAiLink && onRejectAiLink && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white/95 backdrop-blur p-4 rounded-xl shadow-2xl border border-violet-200 z-30 max-w-sm animate-in fade-in zoom-in-95 ring-4 ring-violet-50">
                <div className="flex items-center gap-2 mb-3 text-violet-700 font-bold uppercase text-xs tracking-wider border-b border-violet-100 pb-2">
                    <Sparkles size={14} /> AI Suggested Connection
                </div>
                <div className="flex items-center gap-2 justify-center mb-4 text-sm font-semibold text-slate-800 bg-slate-50 p-2 rounded border border-slate-100">
                    <span className="bg-white px-2 py-1 rounded border shadow-sm">{getId(activeAiLink.source)}</span>
                    <span className="text-slate-400">→</span>
                    <span className="bg-white px-2 py-1 rounded border shadow-sm">{getId(activeAiLink.target)}</span>
                </div>
                <p className="text-xs text-slate-600 mb-4 bg-violet-50 p-3 rounded italic border border-violet-100 leading-relaxed">
                    "{activeAiLink.reason}"
                </p>
                <div className="flex gap-3">
                    <button 
                        onClick={() => onRejectAiLink(normalizeLink(activeAiLink))} 
                        className="flex-1 py-2 text-xs font-bold text-slate-500 bg-slate-100 hover:bg-slate-200 hover:text-red-500 rounded flex items-center justify-center gap-1 transition-colors"
                    >
                        <X size={14} /> Reject
                    </button>
                    <button 
                        onClick={() => onAcceptAiLink(normalizeLink(activeAiLink))} 
                        className="flex-1 py-2 text-xs font-bold text-white bg-violet-600 hover:bg-violet-700 rounded shadow-sm flex items-center justify-center gap-1 transition-colors"
                    >
                        <Check size={14} /> Accept
                    </button>
                </div>
            </div>
        )}

        {/* UI Overlay: Generate AI Links Button */}
        {onGenerateAiLinks && (
            <button 
                onClick={onGenerateAiLinks}
                disabled={isGeneratingAiLinks}
                className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-2 bg-white/90 backdrop-blur text-violet-700 text-xs font-bold uppercase tracking-wide border border-violet-200 rounded-lg shadow-sm hover:shadow-md hover:bg-violet-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {isGeneratingAiLinks ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
                <span>{isGeneratingAiLinks ? 'Thinking...' : 'AI Suggest Links'}</span>
            </button>
        )}

        <div ref={tooltipRef} className="fixed z-50 hidden bg-slate-900/95 text-white p-3 rounded-lg shadow-xl border border-slate-700 backdrop-blur-sm pointer-events-none max-w-[250px] transition-opacity duration-150" />
        <div className="absolute bottom-4 right-4 text-xs text-slate-400 opacity-50 pointer-events-none">Click to select • Double-click to edit • Drag to move</div>
    </div>
  );
};
