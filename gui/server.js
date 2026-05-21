const fs = require("fs");
const path = require("path");
const http = require("http");

const publicDir = __dirname;

loadEnv(path.join(__dirname, ".env"));

const port = Number(process.env.PORT || 3000);
const elevenLabsApiKey = String(process.env.ELEVENLABS_API_KEY || "").trim();
const defaultVoiceId = String(process.env.ELEVENLABS_VOICE_ID || "JBFqnCBsd6RMkjVDRZzb").trim();
const modelId = String(process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2").trim();
const outputFormat = String(process.env.ELEVENLABS_OUTPUT_FORMAT || "mp3_44100_128").trim();
const maxTextLength = Number(process.env.TTS_MAX_TEXT_LENGTH || 2500);

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const publicFiles = new Set(["index.html", "script.js", "styles.css"]);

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

    if (req.method === "GET" && url.pathname === "/api/health") {
      sendJson(res, 200, {
        ok: true,
        elevenLabsConfigured: Boolean(elevenLabsApiKey),
        voiceId: defaultVoiceId,
        modelId,
      });
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/tts") {
      await handleTextToSpeech(req, res);
      return;
    }

    if (req.method === "GET" || req.method === "HEAD") {
      await serveStaticFile(req, res, url.pathname);
      return;
    }

    sendJson(res, 405, { error: "Method not allowed" });
  } catch (error) {
    console.error(error);
    if (!res.headersSent) {
      sendJson(res, 500, { error: "Server error" });
    } else {
      res.end();
    }
  }
});

server.listen(port, () => {
  console.log(`medchat is running at http://localhost:${port}`);
});

async function handleTextToSpeech(req, res) {
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

  res.writeHead(200, {
    "Content-Type": elevenLabsResponse.headers.get("content-type") || "audio/mpeg",
    "Cache-Control": "no-store",
  });

  for await (const chunk of elevenLabsResponse.body) {
    res.write(chunk);
  }
  res.end();
}

async function serveStaticFile(req, res, pathname) {
  const decodedPath = decodeURIComponent(pathname);
  const relativePath = decodedPath === "/" ? "index.html" : decodedPath.replace(/^\/+/, "");
  const filePath = path.resolve(publicDir, relativePath);
  const relativeToPublic = path.relative(publicDir, filePath);
  const publicPath = relativeToPublic.replaceAll(path.sep, "/");

  if (relativeToPublic.startsWith("..") || path.isAbsolute(relativeToPublic)) {
    sendJson(res, 403, { error: "Forbidden" });
    return;
  }

  if (!isPublicFile(publicPath)) {
    sendJson(res, 404, { error: "Not found" });
    return;
  }

  const stat = await fs.promises.stat(filePath).catch(() => null);
  if (!stat?.isFile()) {
    sendJson(res, 404, { error: "Not found" });
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(200, {
    "Content-Type": mimeTypes[ext] || "application/octet-stream",
    "Cache-Control": "no-store",
  });

  if (req.method === "HEAD") {
    res.end();
    return;
  }

  fs.createReadStream(filePath).pipe(res);
}

function isPublicFile(publicPath) {
  if (publicFiles.has(publicPath)) return true;
  if (!publicPath.startsWith("assets/")) return false;
  return !publicPath.split("/").some((part) => part.startsWith("."));
}

function readJsonBody(req) {
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

function loadEnv(filePath) {
  if (!fs.existsSync(filePath)) return;

  const envFile = fs.readFileSync(filePath, "utf8");
  for (const line of envFile.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex === -1) continue;

    const key = trimmed.slice(0, separatorIndex).trim();
    let value = trimmed.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    if (!process.env[key]) process.env[key] = value;
  }
}


