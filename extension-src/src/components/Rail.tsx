import React, { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { LucideIcon } from 'lucide-react';
import {
  MessageSquare, Search, Languages, PenLine, CheckSquare,
  Database, Zap, FileText, Wrench, File, Mic, Radio,
  Camera, Youtube, FlaskConical, Calculator, Terminal, Code2,
  Image, BarChart2, Bot, Cpu, Settings, Crosshair,
  Presentation, Globe,
} from 'lucide-react';
import { useStore } from '../store';
import type { PanelId } from '../types';

/* ------------------------------------------------------------------ */
/*  Category tab definitions                                          */
/* ------------------------------------------------------------------ */

interface SubPanel {
  id: PanelId;
  label: string;
  Icon: LucideIcon;
}

interface CategoryTab {
  key: string;
  label: string;
  Icon: LucideIcon;
  panels: SubPanel[];
}

const CATEGORIES: CategoryTab[] = [
  {
    key: 'chat',
    label: 'Chat',
    Icon: MessageSquare,
    panels: [
      { id: 'chat', label: 'Chat', Icon: MessageSquare },
      { id: 'search', label: 'Search', Icon: Search },
      { id: 'translate', label: 'Translate', Icon: Languages },
      { id: 'ask', label: 'Ask', Icon: Zap },
    ],
  },
  {
    key: 'create',
    label: 'Create',
    Icon: PenLine,
    panels: [
      { id: 'write', label: 'Write', Icon: PenLine },
      { id: 'grammar', label: 'Grammar', Icon: CheckSquare },
      { id: 'artifacts', label: 'Artifacts', Icon: Code2 },
      { id: 'webcreator', label: 'Web Creator', Icon: Globe },
      { id: 'image', label: 'Image', Icon: Image },
      { id: 'code', label: 'Code', Icon: Terminal },
      { id: 'slides', label: 'Slides', Icon: Presentation },
    ],
  },
  {
    key: 'research',
    label: 'Research',
    Icon: FlaskConical,
    panels: [
      { id: 'summary', label: 'Summary', Icon: FileText },
      { id: 'research', label: 'Research', Icon: FlaskConical },
      { id: 'youtube', label: 'YouTube', Icon: Youtube },
      { id: 'pdf', label: 'PDF', Icon: File },
      { id: 'math', label: 'Math', Icon: Calculator },
      { id: 'wisebase', label: 'Wisebase', Icon: Database },
    ],
  },
  {
    key: 'tools',
    label: 'Tools',
    Icon: Wrench,
    panels: [
      { id: 'tools', label: 'Tools', Icon: Wrench },
      { id: 'compare', label: 'Compare', Icon: BarChart2 },
      { id: 'capture', label: 'Capture', Icon: Crosshair },
      { id: 'ocr', label: 'OCR', Icon: Camera },
      { id: 'voice', label: 'Voice', Icon: Mic },
      { id: 'record', label: 'Record', Icon: Radio },
      { id: 'agent', label: 'Agent', Icon: Bot },
    ],
  },
  {
    key: 'settings',
    label: 'Settings',
    Icon: Settings,
    panels: [
      { id: 'models', label: 'Models', Icon: Cpu },
      { id: 'settings', label: 'Settings', Icon: Settings },
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/** Return the category key that owns a given panel id */
function categoryForPanel(panelId: PanelId): string {
  for (const cat of CATEGORIES) {
    if (cat.panels.some((p) => p.id === panelId)) return cat.key;
  }
  return CATEGORIES[0].key;
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function Rail() {
  const { activePanel, setPanel } = useStore();

  // Which tab is currently expanded — auto-follows activePanel's parent
  const [expandedTab, setExpandedTab] = useState<string>(categoryForPanel(activePanel));

  // Sync expanded tab when activePanel changes externally
  useEffect(() => {
    const ownerKey = categoryForPanel(activePanel);
    if (ownerKey !== expandedTab) setExpandedTab(ownerKey);
  }, [activePanel]);

  // Tooltip state (portal-based)
  const [tooltip, setTooltip] = useState<{ label: string; top: number; right: number } | null>(null);

  const showTooltip = useCallback((label: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({ label, top: rect.top + rect.height / 2, right: window.innerWidth - rect.left + 8 });
  }, []);

  const hideTooltip = useCallback(() => setTooltip(null), []);

  const handleTabClick = useCallback(
    (key: string) => {
      if (expandedTab === key) {
        // Already expanded — collapse (optional: keep open, but accordion UX expects toggle)
        // We keep it open since one tab must always be expanded for discoverability
        return;
      }
      setExpandedTab(key);
    },
    [expandedTab],
  );

  return (
    <>
      <nav className="rail-nav">
        <div className="rail-section">
          {CATEGORIES.map((cat) => {
            const isExpanded = expandedTab === cat.key;
            const catOwnsActive = categoryForPanel(activePanel) === cat.key;

            return (
              <React.Fragment key={cat.key}>
                {/* ---- Category tab button ---- */}
                <button
                  onClick={() => handleTabClick(cat.key)}
                  onMouseEnter={(e) => showTooltip(cat.label, e)}
                  onMouseLeave={hideTooltip}
                  title={cat.label}
                  className={`rail-btn rail-icon-btn ${catOwnsActive ? 'rail-active' : ''}`}
                  style={{ width: 36, height: 36 }}
                >
                  <cat.Icon
                    size={18}
                    strokeWidth={catOwnsActive ? 2.2 : 1.5}
                    style={{
                      transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s',
                    }}
                  />
                </button>

                {/* ---- Category label ---- */}
                {isExpanded && (
                  <div className="rail-group-label">{cat.label.toUpperCase()}</div>
                )}

                {/* ---- Expanded sub-panels ---- */}
                {isExpanded && (
                  <div className="rail-extras">
                    {cat.panels.map((panel) => {
                      const isActive = activePanel === panel.id;
                      return (
                        <button
                          key={panel.id}
                          onClick={() => setPanel(panel.id)}
                          onMouseEnter={(e) => showTooltip(panel.label, e)}
                          onMouseLeave={hideTooltip}
                          title={panel.label}
                          className={`rail-btn rail-icon-btn ${isActive ? 'rail-active' : ''}`}
                          style={{ width: 32, height: 32, marginLeft: 4 }}
                        >
                          <panel.Icon
                            size={16}
                            strokeWidth={isActive ? 2.2 : 1.5}
                            style={{
                              transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s',
                            }}
                          />
                        </button>
                      );
                    })}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />
      </nav>

      {/* Tooltip — portaled to body to escape rail overflow clipping */}
      {tooltip &&
        createPortal(
          <div
            className="rail-tooltip"
            style={{
              position: 'fixed',
              right: tooltip.right,
              top: tooltip.top,
              transform: 'translateY(-50%)',
              zIndex: 9999,
            }}
          >
            {tooltip.label}
          </div>,
          document.body,
        )}
    </>
  );
}
