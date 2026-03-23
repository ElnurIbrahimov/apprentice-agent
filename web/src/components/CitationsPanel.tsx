import { useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import type { Message, Citation } from '../types';
import { XMarkIcon, ChevronDownIcon, ArrowTopRightOnSquareIcon } from '@heroicons/react/24/outline';

interface CitationCardProps {
  citation: Citation;
  messageId: string;
  isHighlighted: boolean;
  onHover: (citationId: number | null, messageId: string | null) => void;
}

function CitationCard({ citation, messageId, isHighlighted, onHover }: CitationCardProps) {
  return (
    <a
      href={/^https?:\/\//i.test(citation.url) ? citation.url : '#'}
      target="_blank"
      rel="noopener noreferrer"
      className="block group/card"
      onMouseEnter={() => onHover(citation.id, messageId)}
      onMouseLeave={() => onHover(null, null)}
    >
      <div
        className="flex items-start gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 cursor-pointer"
        style={{
          background: isHighlighted
            ? 'rgba(139, 92, 246, 0.15)'
            : 'rgba(255, 255, 255, 0.03)',
          border: `1px solid ${isHighlighted ? 'rgba(139, 92, 246, 0.4)' : 'rgba(255, 255, 255, 0.06)'}`,
        }}
      >
        {/* Number badge */}
        <span
          className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full text-[11px] font-bold mt-0.5"
          style={{
            background: isHighlighted
              ? 'rgba(139, 92, 246, 0.5)'
              : 'rgba(255, 255, 255, 0.1)',
            color: isHighlighted ? '#e9d5ff' : '#a1a1aa',
          }}
        >
          {citation.id}
        </span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-medium text-gray-200 group-hover/card:text-purple-300 transition-colors truncate">
              {citation.title || citation.url}
            </span>
            <ArrowTopRightOnSquareIcon className="w-3 h-3 text-gray-500 group-hover/card:text-purple-400 flex-shrink-0 transition-colors" />
          </div>
          {citation.snippet && (
            <p className="text-xs text-gray-500 mt-1 line-clamp-2 leading-relaxed">
              {citation.snippet}
            </p>
          )}
          <span className="text-[10px] text-gray-600 mt-1 block truncate">
            {citation.url}
          </span>
        </div>
      </div>
    </a>
  );
}

interface MessageCitationsGroupProps {
  message: Message;
  hoveredCitation: { id: number; messageId: string } | null;
  onHover: (citationId: number | null, messageId: string | null) => void;
}

function MessageCitationsGroup({ message, hoveredCitation, onHover }: MessageCitationsGroupProps) {
  const [collapsed, setCollapsed] = useState(false);
  const citations = message.citations || [];
  if (citations.length === 0) return null;

  // Truncate message content for the group header
  const preview = message.content.length > 60
    ? message.content.slice(0, 60).trim() + '...'
    : message.content;

  return (
    <div className="mb-4">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center gap-2 px-1 py-1.5 text-left text-gray-400 hover:text-gray-200 transition-colors"
      >
        <ChevronDownIcon
          className="w-3 h-3 transition-transform flex-shrink-0"
          style={{ transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)' }}
        />
        <span className="text-xs font-medium truncate">{preview}</span>
        <span className="ml-auto text-[10px] text-gray-600 flex-shrink-0">
          {citations.length} source{citations.length !== 1 ? 's' : ''}
        </span>
      </button>
      {!collapsed && (
        <div className="space-y-1.5 mt-1.5">
          {citations.map((c) => (
            <CitationCard
              key={`${message.id}-${c.id}`}
              citation={c}
              messageId={message.id}
              isHighlighted={
                hoveredCitation !== null &&
                hoveredCitation.id === c.id &&
                hoveredCitation.messageId === message.id
              }
              onHover={onHover}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function CitationsPanel() {
  const messages = useChatStore((s) => s.messages);
  const citationsPanelOpen = useChatStore((s) => s.citationsPanelOpen);
  const setCitationsPanelOpen = useChatStore((s) => s.setCitationsPanelOpen);
  const hoveredCitation = useChatStore((s) => s.hoveredCitation);
  const setHoveredCitation = useChatStore((s) => s.setHoveredCitation);

  const messagesWithCitations = messages.filter(
    (m) => m.citations && m.citations.length > 0 && !m.isStreaming
  );

  const handleHover = useCallback(
    (citationId: number | null, messageId: string | null) => {
      if (citationId !== null && messageId !== null) {
        setHoveredCitation({ id: citationId, messageId });
      } else {
        setHoveredCitation(null);
      }
    },
    [setHoveredCitation]
  );

  if (!citationsPanelOpen) return null;

  return (
    <div
      className="flex flex-col h-full border-l overflow-hidden"
      style={{
        width: 320,
        minWidth: 320,
        background: 'rgba(10, 10, 15, 0.85)',
        borderColor: 'rgba(255, 255, 255, 0.06)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b flex-shrink-0"
        style={{ borderColor: 'rgba(255, 255, 255, 0.06)' }}
      >
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-purple-400" viewBox="0 0 16 16" fill="currentColor">
            <path d="M1 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-1 1H2a1 1 0 01-1-1V3zm0 7a1 1 0 011-1h8a1 1 0 011 1v3a1 1 0 01-1 1H2a1 1 0 01-1-1v-3z" />
          </svg>
          <span className="text-sm font-medium text-gray-200">Sources</span>
          <span
            className="text-[10px] font-medium px-1.5 py-0.5 rounded-full"
            style={{
              background: 'rgba(139, 92, 246, 0.2)',
              color: '#c4b5fd',
            }}
          >
            {messagesWithCitations.reduce((sum, m) => sum + (m.citations?.length || 0), 0)}
          </span>
        </div>
        <button
          onClick={() => setCitationsPanelOpen(false)}
          className="p-1 rounded-md text-gray-500 hover:text-gray-300 hover:bg-white/[0.06] transition-colors"
          aria-label="Close citations panel"
        >
          <XMarkIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        {messagesWithCitations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-600 text-sm text-center px-4">
            <svg className="w-8 h-8 mb-3 text-gray-700" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
            No sources yet.
            <span className="text-xs text-gray-700 mt-1">
              Citations will appear here when AURA references sources.
            </span>
          </div>
        ) : (
          messagesWithCitations.map((msg) => (
            <MessageCitationsGroup
              key={msg.id}
              message={msg}
              hoveredCitation={hoveredCitation}
              onHover={handleHover}
            />
          ))
        )}
      </div>
    </div>
  );
}
