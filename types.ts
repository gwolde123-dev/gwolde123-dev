
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

export interface CourseContent {
    id: string;
    title: string;
    difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
    type: 'Video' | 'Quiz' | 'Interactive';
    subject: string;
}

export interface StudentProgress {
    studentId: string;
    contentId: string;
    score: number;
    timestamp: string;
}

export interface AdaptiveRecommendation {
    studentId: string;
    recommendedContent: CourseContent[];
    reason: string;
}

export interface ModelConfig {
  taskType: 'classification' | 'regression' | 'forecasting';
  algorithm: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  hiddenLayers: number;
  layerSize: number;
  dropout: number;
  activation: 'relu' | 'tanh' | 'sigmoid';
  validationProtocol: 'holdout' | 'k-fold';
  kFoldSplits?: number;
  metrics: string[];
  // Transformer specific
  transformerLayers?: number;
  attentionHeads?: number;
  embeddingDim?: number;
}

export interface TestResult {
  id: string;
  name: string;
  level: 'unit' | 'integration' | 'system';
  category?: string;
  description?: string;
  expectedResult?: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  duration?: string;
  score?: number;
  threshold?: number;
}

export interface SystemEvent {
  id: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

export interface DeploymentTarget {
    platform: 'mobile' | 'web' | 'iot';
    os: 'android' | 'ios' | 'linux';
    buildStatus: 'idle' | 'building' | 'signed' | 'deployed';
    version: string;
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

export interface GraphTestResult {
    name: string;
    passed: boolean;
    details: string;
}
