import { useState, useEffect } from 'react';
import type { TabId } from '../types';
import {
  ChatBubbleLeftRightIcon,
  WrenchScrewdriverIcon,
  ChartBarIcon,
  Cog8ToothIcon,
  CommandLineIcon,
} from '@heroicons/react/24/outline';
import { haptic } from '../utils/haptics';

const MOBILE_TABS: { id: TabId; icon: React.ComponentType<{ className?: string }>; label: string }[] = [
  { id: 'chat', icon: ChatBubbleLeftRightIcon, label: 'Chat' },
  { id: 'create', icon: CommandLineIcon, label: 'Create' },
  { id: 'tools', icon: WrenchScrewdriverIcon, label: 'Tools' },
  { id: 'insights', icon: ChartBarIcon, label: 'Insights' },
  { id: 'settings', icon: Cog8ToothIcon, label: 'Settings' },
];

interface BottomTabBarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  badges?: Partial<Record<TabId, number>>;
}

export function BottomTabBar({ activeTab, onTabChange, badges }: BottomTabBarProps) {
  // Hide when virtual keyboard is open (viewport shrinks significantly)
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const handler = () => {
      // If viewport height is significantly less than window height, keyboard is open
      setHidden(vv.height < window.innerHeight * 0.75);
    };
    vv.addEventListener('resize', handler);
    return () => vv.removeEventListener('resize', handler);
  }, []);

  if (hidden) return null;

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around border-t border-chat-border lg:hidden"
      style={{
        paddingBottom: 'env(safe-area-inset-bottom, 0px)',
        minHeight: 56,
        background: 'var(--surface-1)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
      }}
    >
      {MOBILE_TABS.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => {
              haptic(10);
              if (isActive && tab.id === 'chat') {
                document.dispatchEvent(new CustomEvent('aura:scroll-to-top'));
              }
              onTabChange(tab.id);
            }}
            className={`relative flex flex-col items-center justify-center gap-0.5 flex-1 transition-colors ${
              isActive ? 'text-chat-accent' : 'text-chat-text-secondary'
            }`}
            style={{ minHeight: 44 }}
            aria-label={tab.label}
          >
            <div className="relative">
              <Icon className={`w-5 h-5 ${isActive ? '' : 'opacity-70'}`} />
              {badges?.[tab.id] ? (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-red-500" />
              ) : null}
            </div>
            <span className={`text-[10px] font-medium ${isActive ? '' : 'text-chat-text-tertiary'}`}>{tab.label}</span>
            {isActive && (
              <span className="absolute -bottom-0.5 w-4 h-1 rounded-full bg-chat-accent" />
            )}
          </button>
        );
      })}
    </nav>
  );
}
