
import { GoogleGenAI } from "@google/genai";
import { AISLE_NODES, AISLE_LINKS, AISLE_PROJECT_TARGET, ARCHITECTURAL_QUALITY } from '../constants';
import { AisleNode, AisleLink } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Construct a rich prompt context
const getFrameworkContext = () => {
  return `
    You are an expert Systems Architect and Educational Technologist specializing in AI for accessibility.
    
    You are analyzing the "AISLE Framework" (Ultra-Compact Architecture for Visually Impaired Education in Tigray).
    
    FRAMEWORK CONCEPT:
    The Interactive Graph represents a "Collaborative AI-Driven Learning Platform" where all inputs (UDL, WCAG, CDP) converge through intermediate processes to support the ultimate goal of "Academic Performance". 
    Each node represents a specific technical module or feature set within this platform (e.g., "WebSocket Audio Stream" for Sensory, "Secure Aggregation Server" for Privacy).
    
    FRAMEWORK DATA:
    Target Project: ${AISLE_PROJECT_TARGET}
    
    Architectural Qualities:
    ${JSON.stringify(ARCHITECTURAL_QUALITY)}
    
    Nodes (Components) with Collaborative Platform Features:
    ${JSON.stringify(AISLE_NODES.map(n => ({ 
      id: n.id, 
      group: n.group, 
      label: n.label, 
      description: n.description, 
      details: n.details, 
      orchestration: n.orchestration, // Describes Event -> State choreography
      features: n.features, // Key-Value pairs binding technical implementation to datasets & collaboration tools
      research: n.research // Contains Objectives, RQs, and Dataset references
    })))}
    
    Relationships (Arrows):
    ${JSON.stringify(AISLE_LINKS)}
    
    Key for Arrows:
    g: guides, e: enables, i: informs, c: creates, p: predicts, s: supports, x: contextualizes
  `;
};

export const analyzeFramework = async (prompt: string): Promise<string> => {
  try {
    const context = getFrameworkContext();
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `
        ${context}
        
        USER QUESTION: ${prompt}
        
        INSTRUCTIONS:
        1. Answer strictly based on the provided framework data.
        2. Use professional, academic, yet accessible language.
        3. Format the response with clear headings (Markdown) and bullet points.
        4. Specifically address the node relationships and articulation of the framework.
        5. If the user asks about the interpretation, explain the flow from Input (Blue) -> Process (Green) -> Output (Red).
        6. When performing "Validation", always include a specific section for "PROS" and "CONS" of the current topology.
        7. When analyzing impact, explicitly rate the "Strength of Alignment to Academic Performance" (e.g., High, Moderate, Low) based on the number of converging paths.
        8. If analyzing Data Features, verify that the 'Key' (Technical feature) logically supports the 'Value' (Dataset/Tool) and aligns with the node's goal.
        9. Treat the framework as a functional "Collaborative Platform", discussing how databases, APIs, and stakeholders interact based on the 'features' data.
      `,
    });
    
    return response.text || "No analysis could be generated at this time.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "An error occurred while communicating with the AI service. Please check your API key.";
  }
};

export const suggestMissingLinks = async (currentLinks: AisleLink[]): Promise<AisleLink[]> => {
    try {
        const context = getFrameworkContext();
        const existingPairs = new Set(currentLinks.map(l => `${l.source}-${l.target}`));

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `
                ${context}

                TASK:
                Analyze the nodes and existing connections. Identify conceptually strong relationships that are currently MISSING in the graph structure.
                Focus on 'enables', 'guides', or 'predicts' relationships that improve the logical flow from Input -> Process -> Output.
                
                Do NOT suggest links that already exist.
                
                Return a JSON array of objects with the following structure:
                [
                    {
                        "source": "NodeID",
                        "target": "NodeID",
                        "type": "suggested", 
                        "reason": "A short sentence explaining why this link should exist based on the framework logic."
                    }
                ]
                
                Provide exactly 3 to 5 high-quality suggestions.
                Output ONLY valid JSON.
            `,
            config: {
                responseMimeType: "application/json"
            }
        });

        const text = response.text;
        if (!text) return [];

        const suggestions = JSON.parse(text);
        
        // Post-process to ensure validity
        const validSuggestions: AisleLink[] = suggestions.map((s: any) => ({
            source: s.source,
            target: s.target,
            type: 'suggested',
            label: 'AI',
            isSuggested: true,
            isAiGenerated: true,
            reason: s.reason,
            weight: 0.5
        })).filter((l: AisleLink) => !existingPairs.has(`${l.source}-${l.target}`));

        return validSuggestions;

    } catch (error) {
        console.error("AI Suggestion Error:", error);
        return [];
    }
};
