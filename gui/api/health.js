const elevenLabsApiKey = String(process.env.ELEVENLABS_API_KEY || "").trim();
const defaultVoiceId = String(process.env.ELEVENLABS_VOICE_ID || "JBFqnCBsd6RMkjVDRZzb").trim();
const modelId = String(process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2").trim();

module.exports = function handler(req, res) {
  if (req.method !== "GET" && req.method !== "HEAD") {
    sendJson(res, 405, { error: "Method not allowed" });
    return;
  }

  sendJson(res, 200, {
    ok: true,
    elevenLabsConfigured: Boolean(elevenLabsApiKey),
    voiceId: defaultVoiceId,
    modelId,
  });
};

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}


