
export interface DataQualityMetrics {
  accuracy?: string;
  accuracyBaseline?: string;
  accuracyThreshold?: string;
  completeness?: string;
  completenessBaseline?: string;
  completenessThreshold?: string;
  timeliness?: string;
  timelinessBaseline?: string;
  timelinessThreshold?: string;
  source?: string;
  lastUpdated?: string;
  knownIssues?: string;
}

export interface UserProfile {
  id: string;
  username: string;
  role: 'admin' | 'teacher' | 'student';
  name: string;
  avatar?: string; // Icon name
  biometricEnabled: boolean;
  voiceEnabled: boolean;
  gestureEnabled: boolean;
  preferences?: {
      theme?: 'light' | 'dark';
      notifications?: boolean;
  }
}

export interface ModelConfig {
  taskType: 'classification' | 'regression';
  algorithm: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationProtocol: 'holdout' | 'k-fold';
  kFoldSplits?: number;
  metrics: string[];
}

export interface TestResult {
  id: string;
  name: string;
  level: 'unit' | 'integration' | 'system';
  status: 'pending' | 'running' | 'passed' | 'failed';
  duration?: string;
}

export interface SystemEvent {
  id: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

export interface AisleNode {
  id: string;
  label: string;
  fullLabel?: string;
  group: 'input' | 'process' | 'output' | 'context';
  description: string;
  details: string[]; // Bullet points from the OCR
  category?: string; // e.g., "Design", "Access", "Ethics"
  conceptualRole?: string; // e.g., "The Why", "The What", "The How"
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  research?: {
    objectives?: string[];
    objectiveDescriptions?: Record<string, string>;
    questions?: string[];
    questionDescriptions?: Record<string, string>;
    datasets?: string[];
    datasetDescriptions?: Record<string, string>;
  };
  orchestration?: string;
  stakeholders?: string[]; // 'Student', 'Teacher', 'Developer', 'Policymaker'
  features?: Record<string, string>;
  dataQuality?: Record<string, DataQualityMetrics>;
  simulation?: {
      value: number; // 0 to 100
      isFixed?: boolean; // If true, this is an input variable set by user
      label?: string; // e.g. "Compliance Score", "Adoption Rate"
  };
}

export interface AisleLink {
  source: string;
  target: string;
  type: string; // 'guides', 'enables', 'informs', 'creates', 'predicts', 'supports', 'contextualizes', 'suggested'
  label?: string; // The shorthand (g, e, i, c, p, s, x, ?)
  isOptimization?: boolean;
  isSuggested?: boolean;
  isAiGenerated?: boolean; // True if suggested by LLM
  reason?: string; // AI explanation for the link
  weight?: number; // 0.0 to 1.0, impact strength for simulation
}

export interface AnalysisSection {
  title: string;
  content: string;
}