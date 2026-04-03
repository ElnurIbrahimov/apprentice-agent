// Context Engine types
export type PageContext = 'article' | 'code' | 'media' | 'email' | 'shopping' | 'general';
export type Cadence = 'passive' | 'engaged' | 'active';

export interface ContextSignal {
  type: PageContext;
  accent: string;
  glow: string;
  icon: string;           // SVG markup
  actions: string[];      // ghost bar actions in relevance order
  cadence: Cadence;
  suppressGhostBars: boolean;
  sessionActions: string[];
  readingProgress: number; // 0-1
}

// Pub/Sub
export type ContextListener = (signal: ContextSignal) => void;

export interface ContextStore {
  get(): ContextSignal;
  subscribe(fn: ContextListener): () => void;
  update(partial: Partial<ContextSignal>): void;
}

// FAB
export type FabSide = 'left' | 'right';

export interface FabState {
  side: FabSide;
  offset: number;
  visible: boolean;
  hiddenOnce: boolean;
  hovering: boolean;
  dragging: boolean;
  popoutOpen: boolean;
}

// Ghost Bar
export type GhostBarTarget = 'text' | 'image';

export interface GhostBarState {
  target: GhostBarTarget;
  visible: boolean;
  expanded: boolean;
  anchorRect: DOMRect | null;
  imageElement: HTMLImageElement | null;
}

// Modal
export interface ModalState {
  open: boolean;
  contentType: 'text' | 'image';
  text: string;
  imageUrl: string;
  originRect: DOMRect | null;
}

// Animation
export interface AnimationConfig {
  duration: number;
  easing: string;
  delay?: number;
}

export interface FlowOptions extends AnimationConfig {
  direction: 'up' | 'down';
}

// Module Init
export interface ContentModule {
  init(container: HTMLElement, store: ContextStore, ext: typeof chrome): void;
  destroy?(): void;
}

// Messages (outbound: content → background)
export interface OpenPanelMessage { type: 'OPEN_PANEL'; panel: string; }
export interface OpenWithTextMessage { type: 'OPEN_WITH_TEXT'; action: string; text: string; url: string; title: string; }
export interface SaveKnowledgeMessage { type: 'SAVE_KNOWLEDGE'; text: string; url: string; title: string; }
export interface QuickActionMessage { type: 'QUICK_ACTION'; action: string; text: string; language?: string; threadContext?: string; }
export interface ImageDescribeMessage { type: 'IMAGE_DESCRIBE'; imageUrl: string; }
export interface ImageEditOpenMessage { type: 'IMAGE_EDIT_OPEN'; imageUrl: string; }
export interface ImageSaveMessage { type: 'IMAGE_SAVE'; imageUrl: string; }

export type OutboundMessage =
  | OpenPanelMessage | OpenWithTextMessage | SaveKnowledgeMessage
  | QuickActionMessage | ImageDescribeMessage | ImageEditOpenMessage | ImageSaveMessage;

// Messages (inbound: background → content)
export interface ExtractPageMsg { type: 'EXTRACT_PAGE'; }
export interface GetDomMsg { type: 'GET_DOM'; }
export interface ExecActionMsg { type: 'EXEC_ACTION'; action: ExecActionParams; }
export interface ShowOcrOverlayMsg { type: 'SHOW_OCR_OVERLAY'; dataUrl: string; }
export interface ShowDockMsg { type: 'SHOW_DOCK'; }

export interface ExecActionParams {
  action: 'click' | 'type' | 'scroll' | 'selectOption';
  selector?: string;
  text?: string;
  url?: string;
  amount?: number;
  value?: string;
}

// Highlight
export interface HighlightData {
  id: string;
  url: string;
  text: string;
  xpath: string;
  context: string;
  timestamp: number;
  color: string;
  pageTitle: string;
  stale?: boolean;
}

// FAB Action Items
export interface DockItemDef {
  svg: string;
  action: string;
  tip: string;
}
