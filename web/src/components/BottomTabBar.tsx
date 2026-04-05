import { useState, useEffect } from 'react';
import type { TabId } from '../types';
import {
  ChatBubbleLeftRightIcon,
  WrenchScrewdriverIcon,
  ChartBarIcon,
  Cog8ToothIcon,
  CommandLineIcon,
} from '@heroicons/react/24/outline';
import {
  ChatBubbleLeftRightIcon as ChatSolid,
  WrenchScrewdriverIcon as WrenchSolid,
  ChartBarIcon as ChartSolid,
  Cog8ToothIcon as CogSolid,
  CommandLineIcon as CommandSolid,
} from '@heroicons/react/24/solid';
import { haptic } from '../utils/haptics';

const MOBILE_TABS: { id: TabId; icon: React.ComponentType<{ className?: string }>; iconSolid: React.ComponentType<{ className?: string }>; label: string }[] = [
  { id: 'chat', icon: ChatBubbleLeftRightIcon, iconSolid: ChatSolid, label: 'Chat' },
  { id: 'create', icon: CommandLineIcon, iconSolid: CommandSolid, label: 'Create' },
  { id: 'tools', icon: WrenchScrewdriverIcon, iconSolid: WrenchSolid, label: 'Tools' },
  { id: 'insights', icon: ChartBarIcon, iconSolid: ChartSolid, label: 'Insights' },
  { id: 'settings', icon: Cog8ToothIcon, iconSolid: CogSolid, label: 'Settings' },
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
      setHidden(vv.height < window.innerHeight * 0.75);
    };
    vv.addEventListener('resize', handler);
    return () => vv.removeEventListener('resize', handler);
  }, []);

  if (hidden) return null;

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around border-t lg:hidden"
      style={{
        paddingBottom: 'env(safe-area-inset-bottom, 0px)',
        minHeight: 56,
        background: 'var(--surface-0)',
        borderColor: 'var(--border-subtle)',
        backdropFilter: 'blur(20px) saturate(1.5)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.5)',
      }}
    >
      {MOBILE_TABS.map((tab) => {
        const Icon = tab.icon;
        const IconSolid = tab.iconSolid;
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
            className="relative flex flex-col items-center justify-center gap-0.5 flex-1"
            style={{
              minHeight: 48,
              color: isActive ? 'var(--chat-accent)' : 'var(--text-tertiary)',
              transition: 'color 0.2s ease',
            }}
            aria-label={tab.label}
            aria-current={isActive ? 'page' : undefined}
          >
            {/* Active glow */}
            {isActive && (
              <span
                className="absolute top-1 w-8 h-8 rounded-full opacity-20 pointer-events-none"
                style={{ background: 'var(--chat-accent)', filter: 'blur(10px)' }}
              />
            )}
            <div className="relative" style={{ transform: isActive ? 'scale(1.1)' : 'scale(1)', transition: 'transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)' }}>
              {isActive ? (
                <IconSolid className="w-5 h-5" />
              ) : (
                <Icon className="w-5 h-5" />
              )}
              {badges?.[tab.id] ? (
                <span className="absolute -top-0.5 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 border-2" style={{ borderColor: 'var(--surface-0)' }} />
              ) : null}
            </div>
            <span
              className="text-[10px] font-semibold"
              style={{
                opacity: isActive ? 1 : 0.5,
                transition: 'opacity 0.2s ease',
              }}
            >
              {tab.label}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
