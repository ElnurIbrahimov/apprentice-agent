import { useState, useEffect } from 'react';
import type { Tool, FluxMindStatus, VoiceStatus } from '../types';
import { ArrowPathIcon, WrenchScrewdriverIcon, CpuChipIcon, SpeakerWaveIcon } from '@heroicons/react/24/outline';

export function ToolsPanel() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [fluxmind, setFluxmind] = useState<FluxMindStatus | null>(null);
  const [voice, setVoice] = useState<VoiceStatus | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [toolsRes, fluxRes, voiceRes] = await Promise.all([
        fetch('/api/tools'),
        fetch('/api/fluxmind'),
        fetch('/api/voice'),
      ]);

      if (toolsRes.ok) {
        const data = await toolsRes.json();
        setTools(data.tools || []);
      }
      if (fluxRes.ok) {
        setFluxmind(await fluxRes.json());
      }
      if (voiceRes.ok) {
        setVoice(await voiceRes.json());
      }
    } catch (e) {
      console.error('Failed to fetch tools data:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div className="space-y-4">
      {/* FluxMind */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-chat-text font-medium flex items-center gap-2">
            <CpuChipIcon className="w-5 h-5 text-cyan-400" />
            FluxMind
          </h3>
          <button
            onClick={fetchData}
            className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
            disabled={loading}
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {fluxmind?.enabled ? (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Version</span>
              <span className="text-chat-text">{fluxmind.version}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Accuracy</span>
              <span className="text-green-400">{(fluxmind.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Calibration</span>
              <span className="text-chat-text">{fluxmind.calibration}</span>
            </div>
          </div>
        ) : (
          <div className="text-chat-text-secondary text-sm">FluxMind not available</div>
        )}
      </div>

      {/* Voice */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2 mb-3">
          <SpeakerWaveIcon className="w-5 h-5 text-orange-400" />
          Voice / TTS
        </h3>

        {voice?.available ? (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Engine</span>
              <span className="text-chat-text">{voice.engine}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Sesame</span>
              <span className={voice.sesame_loaded ? 'text-green-400' : 'text-chat-text-secondary'}>
                {voice.sesame_loaded ? 'Loaded' : 'Not loaded'}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-chat-text-secondary text-sm">Voice not available</div>
        )}
      </div>

      {/* Tools List */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2 mb-3">
          <WrenchScrewdriverIcon className="w-5 h-5 text-yellow-400" />
          Available Tools ({tools.length})
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
          {tools.map((tool) => (
            <div
              key={tool.name}
              className="bg-chat-assistant rounded p-2 text-xs"
              title={tool.description}
            >
              <div className="text-chat-text font-medium truncate">{tool.name}</div>
              <div className="text-chat-text-secondary truncate">
                {tool.description?.slice(0, 30)}...
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
