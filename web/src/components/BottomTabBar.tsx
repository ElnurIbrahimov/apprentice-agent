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
}

export function BottomTabBar({ activeTab, onTabChange }: BottomTabBarProps) {
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
        height: 56,
        paddingBottom: 'env(safe-area-inset-bottom, 0px)',
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
              onTabChange(tab.id);
            }}
            className={`flex flex-col items-center justify-center gap-0.5 flex-1 h-full transition-colors ${
              isActive ? 'text-chat-accent' : 'text-chat-text-secondary'
            }`}
            aria-label={tab.label}
          >
            <Icon className="w-5 h-5" />
            {isActive && <span className="text-[9px] font-medium">{tab.label}</span>}
          </button>
        );
      })}
    </nav>
  );
}
