# API Reference

port: 8772

## POST `/synthesize`

**Description:** Synthesize TTS audio and generate subtitles (if configured).

**Endpoint:** `POST /synthesize`

**Headers:**
- `Content-Type: application/json`

**Request body (JSON):**
```json
{
  "text": "Hello world",
  "model": "model_name",      // optional if `api.default_model` is configured
  "font": "03-HurmitNerdFontMono-Regular", // optional, default from config
  "color": "#FFFFFF"         // optional, default from config
}
```

**Responses:**
- `200 OK` - Synthesis accepted
```json
{
  "success": true,
  "text": "Hello world",
  "model": "model_name",
  "font": "03-HurmitNerdFontMono-Regular",
  "color": "#FFFFFF"
}
```
- `400 Bad Request` - Missing or empty `text`, or no model specified and no `default_model` configured
- `500 Internal Server Error` - Server error with `error` message

> **Note:** Default model and subtitle defaults are read from the configuration keys `api.default_model`, `api.default_subtitle_font`, and `api.default_subtitle_color`.

---

## GET `/models`

**Description:** Returns a list of available TTS models.

**Endpoint:** `GET /models`

**Responses:**
- `200 OK`
```json
{ "models": ["model_a", "model_b", "model_c"] }
```
- `500 Internal Server Error` - Server error with `error` message

---

## GET `/status`

**Description:** Returns API status and whether the audio stream is active.

**Endpoint:** `GET /status`

**Response (`200 OK`):**
```json
{
  "status": "running",
  "stream_active": true
}
```

---

## Examples

**cURL - synthesize:**

```bash
curl -X POST http://127.0.0.1:8772/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","model":"model_name"}'
```

**cURL - list models:**

```bash
curl http://127.0.0.1:8772/models
```

**cURL - status:**

```bash
curl http://127.0.0.1:8772/status
```
