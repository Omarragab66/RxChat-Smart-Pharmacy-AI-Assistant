const elevenLabsApiKey = String(process.env.ELEVENLABS_API_KEY || "").trim();
const defaultVoiceId = String(process.env.ELEVENLABS_VOICE_ID || "JBFqnCBsd6RMkjVDRZzb").trim();
const modelId = String(process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2").trim();
const outputFormat = String(process.env.ELEVENLABS_OUTPUT_FORMAT || "mp3_44100_128").trim();
const maxTextLength = Number(process.env.TTS_MAX_TEXT_LENGTH || 2500);

module.exports = async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      sendJson(res, 405, { error: "Method not allowed" });
      return;
    }

    if (!elevenLabsApiKey) {
      sendJson(res, 500, { error: "ElevenLabs API key is not configured" });
      return;
    }

    const body = await readJsonBody(req);
    const text = String(body.text || "").trim();
    const voiceId = String(body.voiceId || defaultVoiceId).trim();

    if (!text) {
      sendJson(res, 400, { error: "Text is required" });
      return;
    }

    if (text.length > maxTextLength) {
      sendJson(res, 413, { error: `Text is too long. Max length is ${maxTextLength} characters.` });
      return;
    }

    if (!/^[A-Za-z0-9_-]+$/.test(voiceId)) {
      sendJson(res, 400, { error: "Invalid ElevenLabs voice id" });
      return;
    }

    const elevenLabsUrl = new URL(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`);
    elevenLabsUrl.searchParams.set("output_format", outputFormat);

    const elevenLabsResponse = await fetch(elevenLabsUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
        "xi-api-key": elevenLabsApiKey,
      },
      body: JSON.stringify({
        text,
        model_id: modelId,
      }),
    });

    if (!elevenLabsResponse.ok) {
      const details = await elevenLabsResponse.text();
      sendJson(res, elevenLabsResponse.status, {
        error: "ElevenLabs request failed",
        details: details.slice(0, 500),
      });
      return;
    }

    const audioBuffer = Buffer.from(await elevenLabsResponse.arrayBuffer());
    res.writeHead(200, {
      "Content-Type": elevenLabsResponse.headers.get("content-type") || "audio/mpeg",
      "Cache-Control": "no-store",
    });
    res.end(audioBuffer);
  } catch (error) {
    console.error(error);
    if (!res.headersSent) {
      sendJson(res, 500, { error: "Server error" });
    } else {
      res.end();
    }
  }
};

function readJsonBody(req) {
  if (req.body !== undefined) {
    if (typeof req.body === "string") {
      return Promise.resolve(req.body ? JSON.parse(req.body) : {});
    }
    return Promise.resolve(req.body || {});
  }

  return new Promise((resolve, reject) => {
    let rawBody = "";

    req.on("data", (chunk) => {
      rawBody += chunk;
      if (rawBody.length > 128 * 1024) {
        req.destroy();
        reject(new Error("Request body is too large"));
      }
    });

    req.on("end", () => {
      if (!rawBody) {
        resolve({});
        return;
      }

      try {
        resolve(JSON.parse(rawBody));
      } catch {
        reject(new Error("Invalid JSON"));
      }
    });

    req.on("error", reject);
  });
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}


