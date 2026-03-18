// Text-to-Speech using Web Speech API

const synth = window.speechSynthesis;
let currentUtterances: SpeechSynthesisUtterance[] = [];
let speakingCallback: (() => void) | null = null;

/** Strip markdown formatting for clean speech output */
function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' code block omitted ')   // fenced code blocks
    .replace(/`([^`]+)`/g, '$1')                           // inline code
    .replace(/!\[.*?\]\(.*?\)/g, ' image ')                // images
    .replace(/\[([^\]]+)\]\(.*?\)/g, '$1')                 // links -> text
    .replace(/^#{1,6}\s+/gm, '')                           // headings
    .replace(/(\*\*|__)(.*?)\1/g, '$2')                    // bold
    .replace(/(\*|_)(.*?)\1/g, '$2')                       // italic
    .replace(/~~(.*?)~~/g, '$1')                           // strikethrough
    .replace(/^\s*[-*+]\s+/gm, '')                         // unordered list markers
    .replace(/^\s*\d+\.\s+/gm, '')                         // ordered list markers
    .replace(/^\s*>\s+/gm, '')                             // blockquotes
    .replace(/---+/g, '')                                  // horizontal rules
    .replace(/\|/g, ' ')                                   // table pipes
    .replace(/\n{2,}/g, '. ')                              // collapse blank lines
    .replace(/\n/g, ' ')                                   // remaining newlines
    .replace(/\s{2,}/g, ' ')                               // collapse whitespace
    .trim();
}

/** Split text on sentence boundaries for natural queuing */
function splitSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by space or end
  const raw = text.match(/[^.!?]+[.!?]+[\s]?|[^.!?]+$/g);
  if (!raw) return [text];
  return raw.map(s => s.trim()).filter(Boolean);
}

/** Pick the best available voice */
function pickVoice(preferred?: string): SpeechSynthesisVoice | null {
  const voices = synth.getVoices();
  if (!voices.length) return null;

  // Exact match by name
  if (preferred) {
    const exact = voices.find(v => v.name === preferred);
    if (exact) return exact;
    const partial = voices.find(v => v.name.toLowerCase().includes(preferred.toLowerCase()));
    if (partial) return partial;
  }

  // Prefer natural-sounding English voices
  const prefs = [
    'Google UK English Female',
    'Google US English',
    'Microsoft Zira',
    'Samantha',
    'Karen',
    'Daniel',
  ];

  for (const name of prefs) {
    const v = voices.find(v => v.name.includes(name));
    if (v) return v;
  }

  // Fallback: first English voice, or first voice overall
  return voices.find(v => v.lang.startsWith('en')) || voices[0];
}

interface SpeakOptions {
  rate?: number;
  pitch?: number;
  voice?: string;
  onEnd?: () => void;
}

/** Speak text aloud, splitting into sentences for long content */
export function speak(text: string, options: SpeakOptions = {}): void {
  stopSpeaking();

  const clean = stripMarkdown(text);
  if (!clean) return;

  const sentences = splitSentences(clean);
  const { rate = 1.0, pitch = 1.0, voice: voiceName, onEnd } = options;
  const selectedVoice = pickVoice(voiceName);

  speakingCallback = onEnd || null;
  currentUtterances = [];

  sentences.forEach((sentence, i) => {
    const utt = new SpeechSynthesisUtterance(sentence);
    utt.rate = rate;
    utt.pitch = pitch;
    if (selectedVoice) utt.voice = selectedVoice;

    if (i === sentences.length - 1) {
      utt.onend = () => {
        currentUtterances = [];
        speakingCallback?.();
        speakingCallback = null;
      };
    }

    utt.onerror = () => {
      currentUtterances = [];
      speakingCallback?.();
      speakingCallback = null;
    };

    currentUtterances.push(utt);
    synth.speak(utt);
  });
}

/** Stop all speech immediately */
export function stopSpeaking(): void {
  synth.cancel();
  currentUtterances = [];
  speakingCallback = null;
}

/** Check if currently speaking */
export function isSpeaking(): boolean {
  return synth.speaking;
}

// Pre-load voices (some browsers load them async)
if (synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = () => { synth.getVoices(); };
}
