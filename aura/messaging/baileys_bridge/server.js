/**
 * Baileys WebSocket Bridge for AURA
 *
 * This Node.js server handles the WhatsApp Web connection via Baileys
 * and exposes it via WebSocket for Python to connect.
 *
 * Install: npm install
 * Run: node server.js
 */

const { default: makeWASocket, DisconnectReason, useMultiFileAuthState } = require('@whiskeysockets/baileys');
const { WebSocketServer } = require('ws');
const qrcode = require('qrcode-terminal');
const path = require('path');
const fs = require('fs');

// Configuration
const WS_PORT = process.env.WS_PORT || 3001;
const AUTH_PATH = path.join(__dirname, '../../data/messaging/whatsapp_session');

// Ensure auth directory exists
if (!fs.existsSync(AUTH_PATH)) {
    fs.mkdirSync(AUTH_PATH, { recursive: true });
}

// WebSocket server for Python connection
const wss = new WebSocketServer({ port: WS_PORT, host: '127.0.0.1' });
let pythonSocket = null;

console.log('');
console.log('='.repeat(50));
console.log('AURA WhatsApp Bridge (Baileys)');
console.log('='.repeat(50));
console.log(`WebSocket server listening on port ${WS_PORT}`);
console.log('Waiting for Python AURA to connect...');
console.log('');

// Handle Python connection
wss.on('connection', (ws, req) => {
    // Verify shared secret if configured
    const expectedSecret = process.env.WS_SECRET;
    if (expectedSecret) {
        const url = new URL(req.url, `http://localhost:${WS_PORT}`);
        const clientSecret = url.searchParams.get('secret');
        if (clientSecret !== expectedSecret) {
            console.log('Rejected connection: invalid secret');
            ws.close(4001, 'Invalid secret');
            return;
        }
    }

    console.log('Python AURA connected!');
    pythonSocket = ws;

    ws.on('message', async (data) => {
        try {
            const message = JSON.parse(data);
            await handlePythonMessage(message);
        } catch (e) {
            console.error('Error handling Python message:', e);
        }
    });

    ws.on('close', () => {
        console.log('Python AURA disconnected');
        pythonSocket = null;
    });

    ws.on('error', (err) => {
        console.error('WebSocket error:', err);
    });
});

// Send to Python
function sendToPython(data) {
    if (pythonSocket && pythonSocket.readyState === 1) {
        pythonSocket.send(JSON.stringify(data));
    }
}

// WhatsApp connection
let sock = null;

async function connectWhatsApp() {
    console.log('Initializing WhatsApp connection...');

    const { state, saveCreds } = await useMultiFileAuthState(AUTH_PATH);

    sock = makeWASocket({
        auth: state,
        printQRInTerminal: false,
        browser: ['AURA', 'Chrome', '120.0.0'],
    });

    // Handle connection events
    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            console.log('');
            console.log('='.repeat(50));
            console.log('Scan this QR code with WhatsApp:');
            console.log('(Settings -> Linked Devices -> Link a Device)');
            console.log('='.repeat(50));
            qrcode.generate(qr, { small: true });
            sendToPython({ type: 'qr', qr: qr });
        }

        if (connection === 'close') {
            const statusCode = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

            console.log('Connection closed, status:', statusCode);
            console.log('Reconnecting:', shouldReconnect);

            sendToPython({ type: 'disconnected', reason: statusCode });

            if (shouldReconnect) {
                setTimeout(connectWhatsApp, 5000);
            } else {
                console.log('');
                console.log('Logged out. Delete the session folder to re-authenticate:');
                console.log(`  rm -rf "${AUTH_PATH}"`);
                console.log('Then restart this server.');
            }
        } else if (connection === 'open') {
            console.log('');
            console.log('='.repeat(50));
            console.log('WhatsApp connected successfully!');
            console.log('='.repeat(50));
            console.log('');
            sendToPython({ type: 'ready' });
        }
    });

    // Save credentials
    sock.ev.on('creds.update', saveCreds);

    // Handle incoming messages
    sock.ev.on('messages.upsert', ({ messages, type }) => {
        // Only process new messages
        if (type !== 'notify') return;

        for (const msg of messages) {
            // Skip messages from self
            if (msg.key.fromMe) continue;

            // Skip if no message content
            if (!msg.message) continue;

            // Extract text content
            const text = msg.message.conversation ||
                        msg.message.extendedTextMessage?.text ||
                        '';

            if (text) {
                console.log(`[Message] ${msg.pushName || 'Unknown'}: ${text.substring(0, 50)}...`);

                sendToPython({
                    type: 'message',
                    id: msg.key.id,
                    from: msg.key.remoteJid,
                    text: text,
                    pushName: msg.pushName || '',
                    timestamp: msg.messageTimestamp
                });
            }
        }
    });
}

// Handle messages from Python
async function handlePythonMessage(message) {
    if (!sock) {
        console.log('WhatsApp not connected, cannot handle message');
        return;
    }

    switch (message.type) {
        case 'send':
            try {
                await sock.sendMessage(message.to, { text: message.text });
                console.log(`[Sent] to ${message.to.split('@')[0]}: ${message.text.substring(0, 50)}...`);
            } catch (e) {
                console.error('Error sending message:', e);
            }
            break;

        case 'typing':
            try {
                await sock.sendPresenceUpdate('composing', message.to);
            } catch (e) {
                // Ignore typing errors
            }
            break;

        case 'read':
            try {
                await sock.readMessages([{ remoteJid: message.to, id: message.id }]);
            } catch (e) {
                // Ignore read errors
            }
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
}

// Start
connectWhatsApp().catch(err => {
    console.error('Failed to start WhatsApp connection:', err);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\nShutting down...');
    if (sock) {
        await sock.logout().catch(() => {});
    }
    wss.close();
    process.exit(0);
});

console.log('Baileys bridge starting...');
