
import { AisleNode, AisleLink, UserProfile } from './types';

// Extracted from the provided OCR and PDF Screenshot
export const AISLE_PROJECT_TARGET = "Multi-sensory AI for visually impaired female/male learners in post-conflict Tigray with low connectivity, using privacy-preserving on-device processing and sustainable educational technology.";

// --- USER MOCK DATA (15 Students, 10 Teachers, 1 Admin) ---
export const MOCK_USERS: UserProfile[] = [
    // Admin
    { id: 'admin_1', username: 'sysadmin', role: 'admin', name: 'Almaz Gebre', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    // Teachers (10)
    { id: 't_1', username: 'teacher_dawit', role: 'teacher', name: 'Dawit Abebe', biometricEnabled: true, voiceEnabled: false, gestureEnabled: false },
    { id: 't_2', username: 'teacher_sara', role: 'teacher', name: 'Sara Tesfaye', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    { id: 't_3', username: 'teacher_hagos', role: 'teacher', name: 'Hagos Berhe', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
    { id: 't_4', username: 'teacher_lety', role: 'teacher', name: 'Letay Girmay', biometricEnabled: true, voiceEnabled: false, gestureEnabled: true },
    { id: 't_5', username: 'teacher_kinfe', role: 'teacher', name: 'Kinfe Michael', biometricEnabled: false, voiceEnabled: false, gestureEnabled: true },
    { id: 't_6', username: 'teacher_lemlem', role: 'teacher', name: 'Lemlem Hailu', biometricEnabled: true, voiceEnabled: true, gestureEnabled: true },
    { id: 't_7', username: 'teacher_yohannes', role: 'teacher', name: 'Yohannes Tekle', biometricEnabled: true, voiceEnabled: false, gestureEnabled: false },
    { id: 't_8', username: 'teacher_helen', role: 'teacher', name: 'Helen Belay', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
    { id: 't_9', username: 'teacher_abrha', role: 'teacher', name: 'Abrha Kahsay', biometricEnabled: true, voiceEnabled: false, gestureEnabled: false },
    { id: 't_10', username: 'teacher_meron', role: 'teacher', name: 'Meron Estifanos', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    // Students (15)
    { id: 's_1', username: 'student_bethel', role: 'student', name: 'Bethelhem Desta', biometricEnabled: true, voiceEnabled: true, gestureEnabled: true },
    { id: 's_2', username: 'student_amanuel', role: 'student', name: 'Amanuel Kebede', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    { id: 's_3', username: 'student_rahwa', role: 'student', name: 'Rahwa Tadesse', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
    { id: 's_4', username: 'student_filmon', role: 'student', name: 'Filmon Araya', biometricEnabled: true, voiceEnabled: false, gestureEnabled: true },
    { id: 's_5', username: 'student_hermon', role: 'student', name: 'Hermon Gide', biometricEnabled: true, voiceEnabled: true, gestureEnabled: true },
    { id: 's_6', username: 'student_selam', role: 'student', name: 'Selamawit Yemane', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
    { id: 's_7', username: 'student_rob', role: 'student', name: 'Robel Assefa', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    { id: 's_8', username: 'student_eden', role: 'student', name: 'Eden Tesfamariam', biometricEnabled: true, voiceEnabled: false, gestureEnabled: true },
    { id: 's_9', username: 'student_daniel', role: 'student', name: 'Daniel Yohannes', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
    { id: 's_10', username: 'student_marta', role: 'student', name: 'Marta Girma', biometricEnabled: true, voiceEnabled: true, gestureEnabled: true },
    { id: 's_11', username: 'student_samuel', role: 'student', name: 'Samuel Berhane', biometricEnabled: true, voiceEnabled: false, gestureEnabled: false },
    { id: 's_12', username: 'student_feven', role: 'student', name: 'Feven Alemu', biometricEnabled: false, voiceEnabled: true, gestureEnabled: true },
    { id: 's_13', username: 'student_natnael', role: 'student', name: 'Natnael Haile', biometricEnabled: true, voiceEnabled: true, gestureEnabled: false },
    { id: 's_14', username: 'student_ludia', role: 'student', name: 'Ludia Tekle', biometricEnabled: true, voiceEnabled: false, gestureEnabled: true },
    { id: 's_15', username: 'student_eyob', role: 'student', name: 'Eyob Solomon', biometricEnabled: false, voiceEnabled: true, gestureEnabled: false },
];

export const AI_ALGORITHMS = [
    { id: 'mlp', name: 'Multi-Layer Perceptron (MLP)', type: 'NN' },
    { id: 'lstm', name: 'Long Short-Term Memory (LSTM)', type: 'RNN' },
    { id: 'cnn', name: 'Convolutional Neural Network (CNN)', type: 'CNN' },
    { id: 'rf', name: 'Random Forest Regressor', type: 'ML' },
    { id: 'xgb', name: 'XGBoost Classifier', type: 'ML' }
];

export const AISLE_NODES: AisleNode[] = [
  // --- INPUT NODES (I - Blue) ---
  {
    id: 'UDL',
    label: 'I: UDL',
    fullLabel: 'Universal Design for Learning',
    group: 'input',
    category: 'Design',
    conceptualRole: 'The Why',
    description: 'Universal Design for Learning - proactive accessibility design. It provides the pedagogical "Why" by ensuring learning is designed for all from the start.',
    details: ['Foundational framework initiating system', 'Proactive accessibility design', 'Provides multiple means of engagement, representation, and action'],
    research: {
        questions: ['RQ2'],
        objectives: ['O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: { 
            'DS01': 'Student Voice/Haptic Survey (Engagement)',
            'DS02': 'Teacher Interview Guidelines (Pedagogy)' 
        },
        questionDescriptions: { 'RQ2': 'To what extent does meta-inclusive design improve academic engagement?' },
        objectiveDescriptions: { 'O3': 'Evaluate pedagogical impact of meta-inclusive design on self-efficacy and engagement.' }
    },
    orchestration: "**Event:** Design Workshop (Figma) -> **State:** Accessible Blueprints. \n\nCollaborative Input: Teachers and VI students co-design interfaces in a shared digital workspace. The system captures these preferences as JSON design tokens to initialize the platform UI.",
    stakeholders: ['Teacher', 'Policymaker', 'Student'],
    features: {
        "Platform Module": "Co-Design Workspace (Figma API)",
        "Data Standard": "JSON Design Tokens",
        "Collaboration": "Real-time Annotation Tools"
    },
    dataQuality: {
        'DS01': {
            accuracy: '92% (Cronbach Alpha > 0.8)',
            accuracyBaseline: '90% (Reliability Std)',
            accuracyThreshold: '80% (Minimum)',
            completeness: '95% (AAPOR RR3)',
            completenessBaseline: '90% (High Quality)',
            completenessThreshold: '80% (Statistical Sig.)',
            timeliness: 'Weekly',
            timelinessBaseline: 'Bi-weekly',
            timelinessThreshold: 'Monthly',
            source: 'Survey (Ref: AAPOR Standards)',
            lastUpdated: '2023-11-01',
            knownIssues: 'Remote district latency'
        }
    },
    simulation: { value: 50, isFixed: true, label: "Design Implementation" }
  },
  {
    id: 'WCAG',
    label: 'I: WCAG',
    fullLabel: 'Web Content Accessibility Guidelines',
    group: 'input',
    category: 'Access Std',
    conceptualRole: 'The What',
    description: 'Technical standards compliance for web accessibility. It represents "The What" by providing the specific technical criteria and success indicators.',
    details: ['Technical standards compliance', 'Foundational framework', 'Perceivable, Operable, Understandable, Robust principles'],
    research: {
        questions: ['RQ2'],
        objectives: ['O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: { 
            'DS01': 'Accessibility Compliance Logs',
            'DS02': 'Teacher Validation of Tools'
        },
        questionDescriptions: { 'RQ2': 'To what extent does meta-inclusive design (using WCAG) improve engagement?' },
        objectiveDescriptions: { 'O3': 'Evaluate pedagogical impact of accessible design standards.' }
    },
    orchestration: "**Event:** Code Commit -> **State:** Validated Interface. \n\nAutomated Pipeline: CI/CD triggers run Axe-core accessibility checks against WCAG 2.1 AA rules before any feature is deployed to the student platform.",
    stakeholders: ['Developer', 'Teacher'],
    features: {
        "Pipeline": "CI/CD Accessibility Gate (Axe-core)",
        "Database": "Compliance Audit Logs (PostgreSQL)",
        "Standard": "WCAG 2.1 AA Ruleset"
    },
    dataQuality: {
        'DS01': {
            accuracy: '100% (Automated)',
            accuracyBaseline: '100% (WCAG 2.1 AA)',
            accuracyThreshold: '98% (Critical Path)',
            completeness: '100% (Full Coverage)',
            completenessBaseline: '100%',
            completenessThreshold: '100%',
            timeliness: 'Real-time (CI/CD)',
            timelinessBaseline: 'Commit-time',
            timelinessThreshold: 'Daily Build',
            source: 'Axe-core (Ref: W3C WCAG)',
            lastUpdated: '2023-11-05',
            knownIssues: 'Manual audit items pending'
        }
    },
    simulation: { value: 80, isFixed: true, label: "Compliance Score" }
  },
  {
    id: 'CDP',
    label: 'I: CDP',
    fullLabel: 'Critical Digital Pedagogy',
    group: 'input',
    category: 'Ethics',
    conceptualRole: 'The How',
    description: 'Ethical and decolonial approaches to pedagogy. It addresses "The How" by examining power structures and ensuring ethical implementation.',
    details: ['Ethical and decolonial approaches', 'Foundational framework', 'Focus on agency and equity'],
    research: {
        questions: ['RQ5'],
        objectives: ['O1', 'O5'],
        datasets: ['DS04', 'DS05'],
        datasetDescriptions: {
            'DS04': 'DPIA & Federated Weights (Restricted)',
            'DS05': 'Open EdTech Platform Data (Zenodo/MIT)'
        },
        objectiveDescriptions: { 
            'O1': 'Construct AI-ready, ethically compliant dataset.',
            'O5': 'Establish open, reusable, DMP-compliant EdTech framework for ethical scaling.'
        },
        questionDescriptions: { 'RQ5': 'How do ethical AI concerns (bias, transparency) uniquely manifest among VI youth?' }
    },
    orchestration: "**Event:** Policy Definition -> **State:** Ethical Guardrails. \n\nGovernance: Defines the Data Protection Impact Assessment (DPIA) logic encoded into the platform's permission system, prioritizing student agency over surveillance.",
    stakeholders: ['Policymaker', 'Teacher'],
    features: {
        "Governance": "Policy Logic Engine",
        "Audit": "Bias Detection Module",
        "Collaboration": "Ethics Review Board Dashboard"
    },
    simulation: { value: 60, isFixed: true, label: "Ethics Adherence" }
  },
  {
    id: 'Tigray',
    label: 'I: Tigray',
    fullLabel: 'Context: Tigray',
    group: 'context',
    category: 'Context',
    description: 'Post-conflict context with low connectivity and high visually impaired female/male enrollment.',
    details: ['Post-conflict context', 'Low connectivity', 'High visually impaired female/male enrollment', 'Contextualizes the entire framework'],
    research: {
        questions: ['RQ1'],
        objectives: ['O1', 'O5'],
        datasets: ['DS05', 'DS01'],
        datasetDescriptions: {
            'DS01': 'Demographic Survey Data',
            'DS05': 'Regional Educational Statistics'
        },
        questionDescriptions: { 'RQ1': 'How do demographic and environmental factors influence access in rural vs. urban Ethiopia?' },
        objectiveDescriptions: { 
            'O1': 'Construct profiles considering demographic/environmental factors.',
            'O5': 'Scale framework to LMICs (Low-to-Middle Income Countries) contexts.'
        }
    },
    stakeholders: ['Policymaker', 'Student', 'Teacher'],
    features: {
        "Constraints": "Network Throttling Simulator",
        "Localization": "Tigrigna NLP Module",
        "Database": "Regional Context Vector Store"
    },
    simulation: { value: 40, isFixed: true, label: "Infrastructure Readiness" }
  },

  // --- INTERMEDIATE NODES (M - Green) ---
  {
    id: 'Sensory',
    label: 'M: Sensory',
    fullLabel: 'Sensory Processing',
    group: 'process',
    category: 'Audio/Haptic',
    description: 'Processing layers with specialized functionality acting as the critical translation layer. It bridges the gap between abstract UDL principles and concrete technical implementation by converting accessibility requirements into multi-modal interfaces (audio, haptic) and code execution.',
    details: ['Processing layer with specialized functionality', 'Multi-modal interfaces', 'Translates UDL into code'],
    research: {
        questions: ['RQ3'],
        objectives: ['O2'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Audio Interaction Logs (Librosa Features)'
        },
        objectiveDescriptions: { 'O2': 'Develop real-time, on-device emotion-aware engine for adaptive multi-sensory resilience interventions.' },
        questionDescriptions: { 'RQ3': 'How can interaction dynamics be modeled using voice and logs?' }
    },
    orchestration: "**Event:** Student Interaction -> **State:** Processed Signal. \n\nData Flow: Captures raw audio via WebSockets and haptic inputs, extracting Librosa features on-device to minimize latency and bandwidth.",
    stakeholders: ['Student', 'Developer'],
    features: {
        "Connectivity": "WebSocket Audio Stream (WSS)",
        "Processing": "Librosa Feature Extractor (Python/WASM)",
        "Output": "Haptic Feedback API"
    },
    dataQuality: {
        'DS01': {
            accuracy: '94% (WER)',
            accuracyBaseline: '95% (State of Art)',
            accuracyThreshold: '85% (Usable)',
            completeness: '99.9% (Packet Loss < 0.1%)',
            completenessBaseline: '99.9%',
            completenessThreshold: '98%',
            timeliness: '120ms (ITU-T G.114)',
            timelinessBaseline: '150ms (Good)',
            timelinessThreshold: '400ms (Limit)',
            source: 'Audio Stream (Ref: ITU-T)',
            lastUpdated: 'Live',
            knownIssues: 'Background noise degradation'
        }
    },
    simulation: { value: 0, label: "Signal Quality" }
  },
  {
    id: 'Meta',
    label: 'M: Meta',
    fullLabel: 'Metacognition',
    group: 'process',
    category: 'Create/Learn',
    description: 'Learning accessibility through accessible content creation.',
    details: ['Accessible content creation', 'Processing layer'],
    research: {
        questions: ['RQ2'],
        objectives: ['O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: {
            'DS01': 'Student Creation Logs',
            'DS02': 'Teacher Workshop Observations'
        },
        questionDescriptions: { 'RQ2': 'To what extent does learning-by-designing improve engagement?' },
        objectiveDescriptions: { 'O3': 'Evaluate pedagogical impact of design-by-doing methodology.' }
    },
    stakeholders: ['Teacher', 'Student'],
    features: {
        "Platform": "Accessible Authoring Tool (React)",
        "Storage": "Content CMS (Strapi/Firebase)",
        "Database": "Student Portfolio DB"
    },
    simulation: { value: 0, label: "Content Accessibility" }
  },
  {
    id: 'Emotion',
    label: 'M: Emotion',
    fullLabel: 'Emotion AI',
    group: 'process',
    category: 'Detect/Adapt',
    description: 'AI-driven frustration detection and adaptive support.',
    details: ['Frustration detection', 'Adaptive support', 'Processing layer'],
    research: {
        questions: ['RQ3'],
        objectives: ['O2'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Voice Sentiment Data & Intervention Logs'
        },
        questionDescriptions: { 'RQ3': 'How can emotional resilience be modeled in real time?' },
        objectiveDescriptions: { 'O2': 'Develop real-time emotion-aware engine (EmotionRegulator).' }
    },
    stakeholders: ['Student', 'Developer'],
    features: {
        "AI Model": "TFLite Emotion Classifier (On-Device)",
        "Input": "Real-time Microphone Stream",
        "Intervention": "Adaptive UI Engine"
    },
    simulation: { value: 0, label: "Detection Accuracy" }
  },
  {
    id: 'Privacy',
    label: 'M: Privacy',
    fullLabel: 'Privacy',
    group: 'process',
    category: 'Device/FL',
    description: 'On-device processing and federated learning (FL).',
    details: ['On-device processing', 'Federated Learning (FL)', 'Processing layer'],
    research: {
        questions: ['RQ6'],
        objectives: ['O4'],
        datasets: ['DS03', 'DS04'],
        datasetDescriptions: {
            'DS03': 'Biometric Facial Data (Closed/Anonymized)',
            'DS04': 'Federated Learning Weight Updates'
        },
        objectiveDescriptions: { 'O4': 'Build privacy-preserving federated AI pipeline compliant with EU AI Act.' },
        questionDescriptions: { 'RQ6': 'Can a federated model be trained on-device without compromising privacy?' }
    },
    orchestration: "**Event:** Model Update -> **State:** Secure Aggregation. \n\nProtocol: Devices compute gradient updates locally and send only encrypted weights to the central aggregator, ensuring raw student data never leaves the device.",
    stakeholders: ['Developer', 'Policymaker'],
    features: {
        "Protocol": "TensorFlow Federated (TFF)",
        "Database": "Local Encrypted Realm DB",
        "Security": "Secure Aggregation Server"
    },
    dataQuality: {
        'DS03': {
            accuracy: 'k=5 (Anonymity)',
            accuracyBaseline: 'k=5 (NIST SP 800-188)',
            accuracyThreshold: 'k=3 (Minimum)',
            completeness: '100% (Encrypted)',
            completenessBaseline: '100%',
            completenessThreshold: '100%',
            timeliness: 'Batch (Epoch)',
            timelinessBaseline: 'Daily',
            timelinessThreshold: 'Weekly',
            source: 'Realm DB (Ref: GDPR/NIST)',
            lastUpdated: '2023-11-04',
            knownIssues: 'Differential privacy noise trade-off'
        }
    },
    simulation: { value: 0, label: "Security Level" }
  },
  {
    id: 'Green',
    label: 'M: Green',
    fullLabel: 'Green Computing',
    group: 'process',
    category: 'Energy/Offline',
    description: 'Energy-efficient and offline-first architecture.',
    details: ['Energy-efficient', 'Offline-first architecture', 'Enables disconnected operation', 'Processing layer'],
    research: {
        questions: ['RQ7'],
        objectives: ['O3', 'O5'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: {
            'DS01': 'Student Sustainability Attitudes Survey',
            'DS02': 'Teacher Infrastructure Reports'
        },
        questionDescriptions: { 'RQ7': 'How does integrating environmental sustainability shape design behaviors?' },
        objectiveDescriptions: { 
            'O3': 'Evaluate sustainability attitudes.',
            'O5': 'Establish reusable, low-resource EdTech framework.'
        }
    },
    orchestration: "**Event:** Network Drop -> **State:** Local Sync. \n\nSync Logic: Service Workers cache learning modules. P2P Mesh capabilities allow devices to sync progress with local peers without internet access.",
    stakeholders: ['Developer', 'Policymaker'],
    features: {
        "Sync": "Opportunistic P2P Mesh",
        "Storage": "IndexedDB / Service Workers",
        "Hardware": "Edge Optimization (Raspberry Pi)"
    },
    simulation: { value: 0, label: "Energy Efficiency" }
  },
  {
    id: 'Efficacy',
    label: 'M: Efficacy',
    fullLabel: 'Self-Efficacy',
    group: 'process',
    category: 'Confidence',
    description: 'Student confidence and creative self-efficacy.',
    details: ['Student confidence', 'Creative self-efficacy', 'Processing layer'],
    research: {
        questions: ['RQ2', 'RQ4'],
        objectives: ['O1', 'O3'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Self-Efficacy Scale (self_efficacy_as_creator)'
        },
        questionDescriptions: { 
            'RQ2': 'Does meta-inclusive design improve self-efficacy?',
            'RQ4': 'What composite psychometric constructs reliably predict outcomes?' 
        },
        objectiveDescriptions: {
            'O1': 'Validate composite constructs like self_efficacy_as_creator.',
            'O3': 'Evaluate impact on self-efficacy.'
        }
    },
    stakeholders: ['Student', 'Teacher'],
    features: {
        "Metric": "Creator Mindset Score",
        "Database": "Psychometric Profile DB",
        "Visualization": "Student Progress Tracker"
    },
    simulation: { value: 0, label: "Student Confidence" }
  },
  {
    id: 'Trust',
    label: 'M: Trust',
    fullLabel: 'Trust',
    group: 'process',
    category: 'AI Rel',
    description: 'User confidence in AI system reliability.',
    details: ['User confidence', 'AI system reliability', 'Processing layer'],
    research: {
        questions: ['RQ5', 'RQ4'],
        objectives: ['O1'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'AI Trust Index (ai_trust_level)'
        },
        questionDescriptions: {
            'RQ5': 'How do ethical concerns affect trust in educational AI?',
            'RQ4': 'Does ai_trust_level predict engagement?'
        },
        objectiveDescriptions: { 'O1': 'Validate ai_trust_level construct.' }
    },
    stakeholders: ['Student', 'Policymaker'],
    features: {
        "Reporting": "Automated Transparency Reports",
        "Feedback": "Trust/Feedback UI Widget",
        "Analysis": "PCA Trust Index Model"
    },
    simulation: { value: 0, label: "System Trust" }
  },
  {
    id: 'Adopt',
    label: 'M: Adopt',
    fullLabel: 'Adoption',
    group: 'process',
    category: 'Tools',
    description: 'Technology integration and teaching method adoption.',
    details: ['Technology integration', 'Teaching method adoption', 'Processing layer'],
    research: {
        questions: ['RQ4'],
        objectives: ['O1', 'O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: {
            'DS01': 'Tech Adoption Index (Rasch Model)',
            'DS02': 'Teacher Usage Logs'
        },
        questionDescriptions: { 'RQ4': 'Does technology_adoption_index predict outcomes?' },
        objectiveDescriptions: { 
            'O1': 'Validate technology_adoption_index.',
            'O3': 'Evaluate engagement and adoption levels.'
        }
    },
    stakeholders: ['Teacher', 'Policymaker'],
    features: {
        "Analytics": "Teacher Adoption Dashboard",
        "Log Analysis": "Rasch Model Logic",
        "Connectivity": "LMS Integration (Moodle/Canvas)"
    },
    simulation: { value: 0, label: "Adoption Rate" }
  },
  {
    id: 'Resil',
    label: 'M: Resil',
    fullLabel: 'Resilience',
    group: 'process',
    category: 'Persist',
    description: 'Learner persistence and adaptive capacity.',
    details: ['Learner persistence', 'Adaptive capacity', 'Processing layer'],
    research: {
        questions: ['RQ3', 'RQ4'],
        objectives: ['O1', 'O2'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Resilience Score (Weighted Mean)'
        },
        questionDescriptions: { 
            'RQ3': 'How can emotional resilience be modeled?',
            'RQ4': 'Does resilience_score predict learning?'
        },
        objectiveDescriptions: { 
            'O1': 'Validate resilience_score composite.',
            'O2': 'Develop adaptive interventions for resilience.'
        }
    },
    stakeholders: ['Student', 'Teacher'],
    features: {
        "AI Agent": "Resilience Intervention Bot",
        "Metric": "Real-time Persistence Score",
        "Database": "Intervention Logs (Encrypted)"
    },
    simulation: { value: 0, label: "Persistence" }
  },

  // --- OUTPUT NODES (O - Red) ---
  {
    id: 'Engage',
    label: 'O: Engage',
    fullLabel: 'Engagement',
    group: 'output',
    category: 'Behavior',
    description: 'Behavioral and emotional participation metrics.',
    details: ['Measurable educational outcome', 'Behavioral participation', 'Emotional participation'],
    research: {
        questions: ['RQ2', 'RQ4'],
        objectives: ['O3'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Engagement Level (engagement_level)'
        },
        questionDescriptions: { 'RQ2': 'Does design improve academic engagement?', 'RQ4': 'Predicting system engagement.' },
        objectiveDescriptions: { 'O3': 'Measure engagement_level.' }
    },
    stakeholders: ['Student', 'Teacher'],
    features: {
        "Metric": "Daily Active Users (DAU)",
        "Dashboard": "Student Engagement Heatmap",
        "Database": "Activity Stream DB"
    },
    simulation: { value: 0, label: "Active Participation" }
  },
  {
    id: 'Know',
    label: 'O: Know',
    fullLabel: 'Knowledge',
    group: 'output',
    category: 'Retain',
    description: 'Knowledge retention and learning persistence.',
    details: ['Measurable educational outcome', 'Knowledge retention'],
    research: {
        questions: ['RQ2'],
        objectives: ['O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: {
            'DS01': 'Pre/Post Knowledge Tests',
            'DS02': 'Teacher Assessment Reports'
        },
        questionDescriptions: { 'RQ2': 'Does design improve knowledge retention?' },
        objectiveDescriptions: { 'O3': 'Evaluate pedagogical impact on knowledge.' }
    },
    stakeholders: ['Student', 'Teacher'],
    features: {
        "Assessment": "Auto-Grading Quiz Module",
        "Database": "Gradebook DB",
        "Analytics": "Knowledge Tracing Model"
    },
    simulation: { value: 0, label: "Retention Rate" }
  },
  {
    id: 'Perform',
    label: 'O: Perform',
    fullLabel: 'Performance',
    group: 'output',
    category: 'Academic',
    description: 'Academic achievement and success outcomes.',
    details: ['Measurable educational outcome', 'Academic achievement', 'Success outcomes'],
    research: {
        questions: ['RQ2', 'RQ4'],
        objectives: ['O1', 'O3'],
        datasets: ['DS01', 'DS02'],
        datasetDescriptions: {
            'DS01': 'Composite Performance Scores',
            'DS02': 'Standardized Test Results'
        },
        questionDescriptions: { 'RQ2': 'Impact on academic outcomes.', 'RQ4': 'Predicting learning outcomes.' },
        objectiveDescriptions: { 'O1': 'Correlate composites with performance.', 'O3': 'Validate improvement in outcomes.' }
    },
    orchestration: "**Event:** Semester End -> **State:** Success Metrics. \n\nAggregation: The platform compiles all upstream metrics (Engagement, Knowledge, Resilience) into a unified 'Academic Performance' report for teachers and policymakers.",
    stakeholders: ['Student', 'Teacher', 'Policymaker'],
    features: {
        "Dashboard": "Master Analytics View",
        "Predictive AI": "Outcome Forecasting Model",
        "Integration": "National Exam Database Sync"
    },
    dataQuality: {
        'DS01': {
            accuracy: '99.9% (Ed-Fi)',
            accuracyBaseline: '99.9% (Transactional)',
            accuracyThreshold: '98% (Audit)',
            completeness: '100% (Roster Sync)',
            completenessBaseline: '100%',
            completenessThreshold: '95%',
            timeliness: 'Real-time',
            timelinessBaseline: 'Daily',
            timelinessThreshold: 'Weekly',
            source: 'LMS (Ref: Ed-Fi Standard)',
            lastUpdated: '2023-11-06',
            knownIssues: 'Offline sync delays'
        }
    },
    simulation: { value: 0, label: "GPA / Success" }
  },
  {
    id: 'Include',
    label: 'O: Include',
    fullLabel: 'Inclusion',
    group: 'output',
    category: 'Social',
    description: 'Social participation and independent learning.',
    details: ['Measurable educational outcome', 'Social participation', 'Independent learning'],
    research: {
        questions: ['RQ1', 'RQ7'],
        objectives: ['O1', 'O5'],
        datasets: ['DS01', 'DS05'],
        datasetDescriptions: {
            'DS01': 'Social Inclusion Survey',
            'DS05': 'Policy Impact Metrics'
        },
        questionDescriptions: { 'RQ1': 'Factors influencing access and inclusion.', 'RQ7': 'Attitudes towards technology.' },
        objectiveDescriptions: { 'O1': 'Profile VI students for inclusion.', 'O5': 'Inclusive framework scaling.' }
    },
    stakeholders: ['Student', 'Policymaker'],
    features: {
        "Metric": "Social Network Analysis",
        "Goal": "SDG 4 Tracking Dashboard",
        "Community": "Peer Support Forum"
    },
    simulation: { value: 0, label: "Inclusion Index" }
  },
  {
    id: 'Sustain',
    label: 'O: Sustain',
    fullLabel: 'Sustainability',
    group: 'output',
    category: 'Environ',
    description: 'Environmental awareness and technology responsibility.',
    details: ['Measurable educational outcome', 'Environmental awareness', 'Technology responsibility'],
    research: {
        questions: ['RQ7'],
        objectives: ['O3', 'O5'],
        datasets: ['DS01'],
        datasetDescriptions: {
            'DS01': 'Sustainability Attitudes Scale'
        },
        questionDescriptions: { 'RQ7': 'Shaping design behaviors and sustainability attitudes.' },
        objectiveDescriptions: { 'O3': 'Measure sustainability_attitudes.', 'O5': 'Sustainable ecosystem.' }
    },
    stakeholders: ['Student', 'Policymaker'],
    features: {
        "Metric": "Carbon Footprint Calculator",
        "Report": "Sustainability Impact Report",
        "IoT": "Device Energy Monitor"
    },
    simulation: { value: 0, label: "Env. Impact" }
  },
];

export const AISLE_LINKS: AisleLink[] = [
  // Input -> Process
  { source: 'UDL', target: 'Sensory', type: 'guides', label: 'g', weight: 0.8 },
  { source: 'WCAG', target: 'Meta', type: 'guides', label: 'g', weight: 0.9 },
  { source: 'CDP', target: 'Emotion', type: 'guides', label: 'g', weight: 0.7 },

  // Process -> Process (Inter-layer)
  { source: 'Sensory', target: 'Privacy', type: 'enables', label: 'e', weight: 0.8 }, 
  { source: 'Meta', target: 'Privacy', type: 'creates', label: 'c', weight: 0.7 },
  { source: 'Emotion', target: 'Green', type: 'creates', label: 'c', weight: 0.6 }, 

  { source: 'Privacy', target: 'Efficacy', type: 'creates', label: 'c', weight: 0.7 },
  { source: 'Privacy', target: 'Trust', type: 'creates', label: 'c', weight: 0.8 },
  { source: 'Meta', target: 'Adopt', type: 'creates', label: 'c', weight: 0.9 },
  { source: 'Emotion', target: 'Resil', type: 'creates', label: 'c', weight: 0.8 },
  
  // OPTIMIZATION: Connecting 'Disconnected' Green Node (Validated)
  { source: 'Green', target: 'Sustain', type: 'creates', label: 'c', isOptimization: true, weight: 0.8 }, 
  { source: 'Green', target: 'Adopt', type: 'enables', label: 'e', isOptimization: true, weight: 0.6 },   
  
  // Convergence for Collaborative Platform
  { source: 'Sensory', target: 'Adopt', type: 'enables', label: 'e', weight: 0.7 }, 
  { source: 'Emotion', target: 'Efficacy', type: 'supports', label: 's', weight: 0.5 }, 
  
  { source: 'Efficacy', target: 'Trust', type: 'supports', label: 's', weight: 0.6 }, 
  
  // Process -> Output
  { source: 'Trust', target: 'Engage', type: 'predicts', label: 'p', weight: 0.8 }, 
  { source: 'Trust', target: 'Know', type: 'predicts', label: 'p', weight: 0.7 },
  
  // ACADEMIC ALIGNMENT: Convergence on Performance
  { source: 'Adopt', target: 'Perform', type: 'predicts', label: 'p', weight: 0.9 }, 
  { source: 'Efficacy', target: 'Perform', type: 'predicts', label: 'p', weight: 0.7 }, 
  { source: 'Resil', target: 'Perform', type: 'predicts', label: 'p', weight: 0.6 }, 
  { source: 'Trust', target: 'Perform', type: 'predicts', label: 'p', weight: 0.5 }, 
  
  // Optimization: Adopt leads to Knowledge
  { source: 'Adopt', target: 'Know', type: 'enables', label: 'e', isOptimization: true, weight: 0.8 },

  { source: 'Resil', target: 'Include', type: 'predicts', label: 'p', weight: 0.7 },
  { source: 'Resil', target: 'Sustain', type: 'supports', label: 's', weight: 0.4 }, 
  
  // PERFORMANCE OPTIMIZATIONS: Ensuring all roads lead to Perform
  { source: 'Engage', target: 'Perform', type: 'predicts', label: 'p', isOptimization: true, weight: 0.8 }, // Optimization: Engagement feeds Performance
  { source: 'Know', target: 'Perform', type: 'enables', label: 'e', isOptimization: true, weight: 0.9 },    // Optimization: Knowledge feeds Performance

  // Contextual interactions
  { source: 'Tigray', target: 'Sensory', type: 'contextualizes', label: 'x', weight: 0.5 },
  { source: 'Tigray', target: 'Green', type: 'contextualizes', label: 'x', weight: 0.5 },
  { source: 'Tigray', target: 'Resil', type: 'contextualizes', label: 'x', weight: 0.5 },
];

export const ARCHITECTURAL_QUALITY = [
  { title: "Loose Coupling", desc: "Independent modules with clear interfaces" },
  { title: "High Cohesion", desc: "Each node has focused, specialized functionality" },
  { title: "Clear Data Flow", desc: "Input → Intermediate → Output transformation" },
  { title: "Contextual Adaptation", desc: "Tigray-specific constraints shape all components" }
];

export const CSV_BRAILLE_BLIND = `Person_ID,Age,Gender,Blindness_Type,Braille_Proficiency_Level,Braille_Letter,English_Letter,Years_of_Using_Braille,Assistive_Technology_Used,Preferred_Reading_Medium,Daily_Braille_Usage,Access_to_Educational_Resources,Occupation,Country,Challenges_Faced,Satisfaction_With_Braille_Tools,Braille_Learning_Resources
1,42,Non-binary,Acquired,Beginner,⠇,H,0,Braille display,Braille,10,Yes,Unemployed,India,Limited resources,5,Online courses
2,38,Prefer not to say,Acquired,Beginner,⠊,K,19,Talking devices,Screen Readers,23,Yes,Retired,France,Assistive technology issues,2,Online courses
3,40,Prefer not to say,Congenital,Advanced,⠍,N,5,Screen reader,Audiobooks,189,No,Unemployed,Canada,Limited resources,4,Physical books
4,50,Non-binary,Acquired,Beginner,⠌,M,44,Screen reader,Screen Readers,94,Yes,Student,USA,Limited resources,2,Online courses
5,53,Female,Acquired,Intermediate,⠆,G,15,Braille display,Braille,166,No,Retired,France,Limited resources,2,Physical books
6,49,Male,Congenital,Beginner,⠄,E,0,Screen reader,Audiobooks,48,Yes,Retired,Germany,Difficulty learning,2,Physical books
7,11,Prefer not to say,Congenital,Beginner,⠆,G,1,Talking devices,Others,115,No,Retired,Australia,Difficulty learning,4,Physical books
8,35,Prefer not to say,Acquired,Beginner,⠄,E,27,Screen reader,Audiobooks,117,Yes,Student,Australia,Assistive technology issues,5,Braille tutor
9,64,Non-binary,Acquired,Intermediate,⠗,X,48,Braille display,Others,212,No,Unemployed,Australia,Limited resources,2,Physical books
10,16,Male,Acquired,Intermediate,⠋,L,7,Talking devices,Others,28,No,Unemployed,Australia,Limited resources,1,Others`;

export const CSV_CONTENT = `content_id,title,subject,content_type,difficulty_level,duration_minutes,description,prerequisites,tags
c_101,Introduction to Algebra,Mathematics,Video,Beginner,15,Basic concepts of algebra.,None,"algebra,math"
c_102,The Scientific Method,Science,Article,Beginner,10,Learn how to think like a scientist.,None,"science,methodology"
c_103,World War II History,History,Video,Intermediate,25,A detailed overview of WWII.,c_102,"history,wwii"
c_104,Poetry Analysis,English,Text,Intermediate,20,How to analyze poetry for deeper meaning.,None,"english,poetry"
c_105,Interactive Science Quiz,Science,Quiz,Beginner,5,Test your knowledge on basic science facts.,None,"science,quiz,interactive"
c_106,Advanced Calculus,Mathematics,Video,Advanced,45,Complex concepts in calculus.,c_101,"calculus,math"
c_107,Introduction to Python,Computer Science,Interactive,Beginner,60,Learn the basics of coding with Python.,None,"coding,python,interactive"
c_108,Biology Lab Simulation,Biology,Interactive,Intermediate,30,Virtual lab for cell biology.,c_102,"biology,lab,interactive"
c_109,The Romantic Period,English,Text,Advanced,35,A deep dive into Romantic literature.,c_104,"english,literature"`;

export const CSV_PROGRESS = `student_id,content_id,score,time_spent,timestamp,student_name,title,subject,difficulty_level
student_10,c_106,76.0,40,2025-10-10 22:00:51.895986,Nandi Ndlovu,Advanced Calculus,Mathematics,Advanced
student_10,c_107,81.0,41,2025-10-10 22:00:51.895949,Nandi Ndlovu,Introduction to Python,Computer Science,Beginner
student_7,c_103,59.0,36,2025-10-10 22:00:51.894745,Thando Mthembu,World War II History,History,Intermediate
student_6,c_109,85.0,54,2025-10-10 16:00:51.894199,Glad Maimele,The Romantic Period,English,Advanced
student_9,c_109,65.0,57,2025-10-10 09:00:51.895458,Sipho Dlamini,The Romantic Period,English,Advanced
student_8,c_109,67.0,8,2025-10-09 21:00:51.895154,Lebo Moloi,The Romantic Period,English,Advanced
student_9,c_106,78.0,39,2025-10-09 18:00:51.895702,Sipho Dlamini,Advanced Calculus,Mathematics,Advanced
student_10,c_105,78.0,5,2025-10-09 11:00:51.895783,Nandi Ndlovu,Interactive Science Quiz,Science,Beginner
student_2,c_106,96.0,41,2025-10-09 00:00:51.892724,Vutlhari Masinga,Advanced Calculus,Mathematics,Advanced
student_6,c_108,66.0,26,2025-10-08 18:00:51.894501,Glad Maimele,Biology Lab Simulation,Biology,Intermediate`;

export const CSV_STUDENTS = `student_id,name,grade_level,learning_style,preferred_subjects
student_1,Bethuel Sonyoni,Grade 10,Visual,"Mathematics, Science"
student_2,Vutlhari Masinga,Grade 9,Auditory,"English, History"
student_3,Siyabonga Hlope,Grade 11,Kinesthetic,"Science, Art"
student_4,Dimbanyika Phindulo,Grade 10,Reading/Writing,"Mathematics, English"
student_5,Andile Maluleke,Grade 9,Visual,"Art, History"
student_6,Glad Maimele,Grade 12,Auditory,"Physics, Chemistry"
student_7,Thando Mthembu,Grade 11,Kinesthetic,"Biology, Geography"
student_8,Lebo Moloi,Grade 10,Reading/Writing,"Literature, Drama"
student_9,Sipho Dlamini,Grade 9,Visual,"Computer Science, Robotics"
student_10,Nandi Ndlovu,Grade 12,Auditory,"Music, History"`;

export const CSV_EMOTION_MULTIMODAL = `image_id,caption_text,emotion_label,resilience_score,efficacy_score,behavior_tag
img_001,"Student smiling while using tactile map","Happy",0.85,0.78,"Engaged"
img_002,"User frowning at audio interface delay","Frustrated",0.42,0.55,"Disengaged"
img_003,"Group of students collaborating on tablet","Excited",0.91,0.88,"Collaborative"
img_004,"Student focusing intently on braille display","Focused",0.76,0.82,"Persisting"
img_005,"User looking confused at screen reader output","Confused",0.35,0.40,"Struggling"
img_006,"Teacher guiding student hand on haptic device","Supported",0.80,0.75,"Mentored"
img_007,"Student laughing after solving math problem","Joyful",0.95,0.92,"Success"
img_008,"User sighing and leaning back from desk","Bored",0.25,0.30,"Passive"
img_009,"Student explaining concept to peer","Confident",0.88,0.90,"Teaching"
img_010,"User repeatedly pressing button with force","Angry",0.20,0.45,"Frustrated"`;
