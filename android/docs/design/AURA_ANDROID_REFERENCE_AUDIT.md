# Aura Android Mobile Reference Audit

Date: 2026-07-13

This audit extracts transferable mobile patterns from official product sources and the actually rendered Aura Web UI. It does not copy competitor branding or desktop layouts.

## 1. Aura Web — local source of truth

Verified reference files:

- `D:/Aura/web/src/components/MessageBubble.tsx`
- `D:/Aura/web/src/components/MessageInput.tsx`
- `D:/Aura/web/src/components/ChatContainer.tsx`
- `D:/Aura/web/src/components/AuraBreathingAvatar.tsx`
- `D:/Aura/web/src/index.css`
- `D:/Aura/web/tailwind.config.js`
- Android evidence: `.hermes/ui-real-audit-sheet.png`

Transferable language:

- compact Aura logomark and role label;
- restrained charcoal/teal palette;
- assistant content without a heavy enclosing bubble;
- asymmetric user bubble with constrained width;
- compact top bar with four high-frequency controls;
- quick prompts adjacent to the composer, not a giant card dashboard;
- deliberate display type for short moments and a neutral UI face elsewhere.

Do not transfer:

- the desktop three-pane shell;
- 768px desktop composition as literal phone geometry;
- desktop sidebar density;
- mouse-first hover affordances.

## 2. ChatGPT Android — integrated conversation modalities

Official sources:

- https://help.openai.com/en/articles/20001274-chatgpt-voice
- https://help.openai.com/en/articles/8400625
- https://help.openai.com/en/articles/8142208-chatgpt-android-app-faq

Observed official patterns:

- Voice may remain inside the main conversation rather than forcing a disconnected experience.
- Distinct voice modes are selected in Settings and have explicit capability differences.
- Spoken responses can appear as text while audio plays.
- Captions, camera, image, and screen-sharing actions are contextual to voice mode.
- Conversation history remains the durable record after voice ends.

Adopt for Aura:

- keep voice transcript and resulting response in the same Chat timeline;
- label dictation, hold-to-talk, and continuous voice explicitly;
- keep secondary camera/file/voice capabilities contextual rather than crowding the default composer.

Avoid:

- three unlabeled microphone gestures that users must learn by accident;
- a voice overlay that hides the durable conversation state.

## 3. Claude Android — clear top-level model and voice placement

Official sources:

- https://www.anthropic.com/news/android-app
- https://support.anthropic.com/en/articles/10065434-using-dictation-on-the-claude-mobile-apps
- https://support.anthropic.com/en/articles/11101966-using-voice-mode-on-claude-mobile-apps
- https://support.anthropic.com/en/articles/8114491-getting-started-with-claude

Observed official patterns:

- The active model is displayed at the top of the mobile screen.
- Dictation is a visible microphone action on the right side of the input.
- Voice mode has explicit in-mode settings and a clearly labeled Stop action.
- Cross-device chat continuity is central rather than hidden.

Adopt for Aura:

- keep a compact model pill at the top of Chat;
- use explicit **Stop**, **Cancel**, and mode names;
- keep History/model context visible but move low-frequency actions to overflow.

Avoid:

- displaying a model name before Aura has verified that the model exists;
- allowing long IDs to consume the entire toolbar.

## 4. Gemini Android — purposeful expressive motion and transparent agents

Official sources:

- https://blog.google/products-and-platforms/products/gemini/new-gemini-app-updates-android/
- https://blog.google/products/gemini/gemini-live-android-tips/
- https://blog.google/innovation-and-ai/products/gemini-app/next-evolution-gemini-app/
- https://blog.google/innovation-and-ai/products/gemini-app/android-multi-step-tasks/

Observed official patterns:

- Live conversation is integrated with camera/screen context.
- The current design language uses typography, color, haptics, and fluid but purposeful motion.
- Long-running work exposes progress, lets the user inspect it, and remains interruptible.
- Background work reports status through notifications rather than silently completing.

Adopt for Aura:

- use motion to communicate listening/thinking/streaming/tool state;
- make tool/hand/proactive progress visible and interruptible;
- stop animation when inactive or offscreen;
- preserve user control during background and multi-step work.

Avoid:

- ornamental infinite animation;
- progress that vanishes without a result or history entry.

## 5. Perplexity Android — mobile-native action transparency

Official sources:

- https://www.perplexity.ai/help-center/en/articles/10450852-how-to-use-the-perplexity-android-assistant
- https://www.perplexity.ai/hub/blog/comet-for-android-is-here
- https://comet-help.perplexity.ai/en/articles/12875447-comet-for-android-quick-start-and-key-features

Observed official patterns:

- The assistant is available through direct Android entry points, not only deep navigation.
- Voice is a first-class mobile interaction.
- Multi-step actions expose what the assistant is doing and allow intervention.
- The Android product was deliberately redesigned for mobile rather than forcing a desktop experience onto a smaller screen.

Adopt for Aura:

- keep Quick Ask/widget/voice as coherent secondary entry points;
- show live tool/hand progress and allow stop/retry;
- preserve a mobile-native four-destination shell while exposing secondary capabilities through contextual navigation.

Avoid:

- treating every feature as an equal Home tile;
- copying desktop navigation or information density.

## 6. Consolidated product decisions

| Area | Aura decision |
|---|---|
| Model selector | Compact top pill; truthful state; search/group/status in content-adaptive sheet |
| Composer | One compact unit; contextual media/voice; 52–144dp height |
| Voice | Three explicit modes; transcript remains in Chat; clear stop/cancel |
| Messages | Assistant content-first, restrained containers; user bubble constrained |
| Tool/agent work | Visible progress, interruptible, durable result/history |
| Home | One primary Ask Aura action plus current priority; secondary destinations compact |
| Navigation | Four persistent top-level destinations, phone-native proportions |
| Motion | Purposeful state communication, no idle offscreen animation |
| Themes | Full semantic light/dark parity; no forced-dark escape hatch |
| Recovery | Every empty/error/no-provider state offers the nearest useful action |

## 7. Explicit anti-copy rules

- Do not reproduce another product’s color palette, icons, copy, or brand motion.
- Do not add capabilities solely because a competitor has them.
- Do not trade Aura’s memory, tools, hands, proactive, or graph identity for a generic chat clone.
- Use references to set interaction quality and proportion—not to erase Aura’s character.
