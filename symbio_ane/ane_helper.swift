// symbio-ane: the side models Symbio runs on the Apple Neural Engine.
//
// The 14B headmaster owns the GPU (MLX) and most of the memory. OCR, a
// vision pass over a screenshot, and small routing decisions do not need it:
// Apple's Vision framework runs its text recognizer and detectors on the
// Neural Engine, and the on-device Apple Intelligence model (FoundationModels)
// can answer a structured routing question. Running them here keeps them off
// the GPU the model is generating on.
//
// One JSON object per line on stdin, one per line on stdout:
//   {"op": "status"}
//   {"op": "ocr", "path": "/tmp/shot.png", "level": "accurate"|"fast"}
//   {"op": "vision", "path": "/tmp/shot.png", "tasks": ["rectangles", "saliency", "classify"]}
//   {"op": "decide", "message": "...", "options": {...}}
// Every reply carries "ok", "ms", and for Vision work "devices": the compute
// device each request was pinned to, so "it ran on the ANE" is reported, not
// assumed.

import CoreGraphics
import CoreML
import Foundation
import ImageIO
import NaturalLanguage
import Vision

#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - plumbing

func reply(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: []),
          let line = String(data: data, encoding: .utf8) else {
        print("{\"ok\":false,\"error\":\"unencodable reply\"}")
        fflush(stdout)
        return
    }
    print(line)
    fflush(stdout)
}

func loadImage(_ path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "symbio-ane", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "cannot read image at \(path)"])
    }
    return image
}

/// Pin every stage of a Vision request to the Neural Engine when the request
/// supports it, and say where each stage went.
func preferNeuralEngine(_ request: VNRequest) -> [String: String] {
    var chosen: [String: String] = [:]
    guard let stages = try? request.supportedComputeStageDevices else { return chosen }
    for (stage, devices) in stages {
        let ane = devices.first { device in
            if case .neuralEngine = device { return true }
            return false
        }
        if let ane = ane {
            request.setComputeDevice(ane, for: stage)
            chosen[stage.rawValue] = "neuralEngine"
        } else if let first = devices.first {
            switch first {
            case .cpu: chosen[stage.rawValue] = "cpu"
            case .gpu: chosen[stage.rawValue] = "gpu"
            case .neuralEngine: chosen[stage.rawValue] = "neuralEngine"
            @unknown default: chosen[stage.rawValue] = "unknown"
            }
        }
    }
    return chosen
}

/// Vision boxes are normalised with the origin at the bottom left; the rest
/// of Symbio (screenshots, clicks) works in pixels from the top left.
func pixelBox(_ box: CGRect, _ width: Int, _ height: Int) -> [Int] {
    let w = Double(width), h = Double(height)
    return [Int((box.minX * w).rounded()), Int(((1 - box.maxY) * h).rounded()),
            Int((box.width * w).rounded()), Int((box.height * h).rounded())]
}

func millis(since start: DispatchTime) -> Int {
    Int((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
}

// MARK: - OCR

/// A region of the screenshot, optionally enlarged: the recognizer misses
/// short labels in small type that it reads fine at 2-3x. Returns the image
/// and how to map its pixels back (offset, scale).
func region(_ image: CGImage, _ args: [String: Any]) -> (CGImage, CGPoint, Double) {
    var cropped = image
    var origin = CGPoint.zero
    if let crop = args["crop"] as? [Int], crop.count == 4 {
        let rect = CGRect(x: crop[0], y: crop[1], width: crop[2], height: crop[3])
            .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        if !rect.isEmpty, let piece = image.cropping(to: rect) {
            cropped = piece
            origin = rect.origin
        }
    }
    let scale = max(1.0, min(4.0, (args["upscale"] as? Double) ?? 1.0))
    if scale > 1.0 {
        let width = Int(Double(cropped.width) * scale), height = Int(Double(cropped.height) * scale)
        if let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                   bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
            context.interpolationQuality = .high
            context.draw(cropped, in: CGRect(x: 0, y: 0, width: width, height: height))
            if let big = context.makeImage() { return (big, origin, scale) }
        }
    }
    return (cropped, origin, 1.0)
}

func ocr(_ args: [String: Any]) throws -> [String: Any] {
    let start = DispatchTime.now()
    let full = try loadImage(args["path"] as? String ?? "")
    let (image, origin, scale) = region(full, args)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = (args["level"] as? String) == "fast" ? .fast : .accurate
    // Off by default: UI labels, file names and code are not dictionary words,
    // and correction "fixes" them into ones that are.
    request.usesLanguageCorrection = (args["correct"] as? Bool) ?? false
    if let languages = args["languages"] as? [String] { request.recognitionLanguages = languages }
    let devices = preferNeuralEngine(request)
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let lines: [[String: Any]] = (request.results ?? []).compactMap { observation in
        guard let best = observation.topCandidates(1).first else { return nil }
        // Back into the full screenshot's pixels, whatever was cropped or enlarged.
        let box = pixelBox(observation.boundingBox, image.width, image.height)
        let mapped = [Int(Double(box[0]) / scale) + Int(origin.x), Int(Double(box[1]) / scale) + Int(origin.y),
                      Int(Double(box[2]) / scale), Int(Double(box[3]) / scale)]
        return ["text": best.string, "conf": Double(best.confidence), "box": mapped]
    }
    return ["ok": true, "lines": lines, "size": [full.width, full.height],
            "devices": devices, "ms": millis(since: start)]
}

// MARK: - vision

func vision(_ args: [String: Any]) throws -> [String: Any] {
    let start = DispatchTime.now()
    let image = try loadImage(args["path"] as? String ?? "")
    let tasks = Set((args["tasks"] as? [String]) ?? ["rectangles", "saliency", "classify"])
    var requests: [VNRequest] = []
    var devices: [String: [String: String]] = [:]

    let rectangles = VNDetectRectanglesRequest()
    rectangles.maximumObservations = (args["max"] as? Int) ?? 64
    rectangles.minimumSize = Float((args["min_size"] as? Double) ?? 0.01)
    rectangles.minimumAspectRatio = 0.05
    rectangles.quadratureTolerance = 20
    let saliency = VNGenerateObjectnessBasedSaliencyImageRequest()
    let classify = VNClassifyImageRequest()
    for (name, request) in [("rectangles", rectangles as VNRequest),
                            ("saliency", saliency as VNRequest),
                            ("classify", classify as VNRequest)] where tasks.contains(name) {
        devices[name] = preferNeuralEngine(request)
        requests.append(request)
    }
    try VNImageRequestHandler(cgImage: image, options: [:]).perform(requests)

    var out: [String: Any] = ["ok": true, "size": [image.width, image.height], "devices": devices]
    if tasks.contains("rectangles") {
        out["rectangles"] = (rectangles.results ?? []).map {
            ["box": pixelBox($0.boundingBox, image.width, image.height), "conf": Double($0.confidence)]
        }
    }
    if tasks.contains("saliency") {
        let objects = (saliency.results?.first?.salientObjects ?? [])
        out["salient"] = objects.map {
            ["box": pixelBox($0.boundingBox, image.width, image.height), "conf": Double($0.confidence)]
        }
    }
    if tasks.contains("classify") {
        let minimum = Float((args["min_conf"] as? Double) ?? 0.15)
        out["labels"] = (classify.results ?? []).filter { $0.confidence >= minimum }.prefix(10).map {
            ["label": $0.identifier, "conf": Double($0.confidence)]
        }
    }
    out["ms"] = millis(since: start)
    return out
}

// MARK: - sentence embeddings (the decision model's input)

/// Apple's on-device contextual (transformer) embedding, mean-pooled over the
/// tokens and L2-normalised: a 512-number fingerprint of what a message is
/// about. Loaded once, then ~7 ms per sentence on the M4. It needs no Apple
/// Intelligence, so the routing decision works on every Mac.
nonisolated(unsafe) var contextualEmbedding: NLContextualEmbedding? = nil

func embed(_ args: [String: Any]) async throws -> [String: Any] {
    let start = DispatchTime.now()
    if contextualEmbedding == nil {
        guard let model = NLContextualEmbedding(language: .english) else {
            return ["ok": false, "error": "no English contextual embedding on this Mac"]
        }
        if !model.hasAvailableAssets { _ = try await model.requestAssets() }
        try model.load()
        contextualEmbedding = model
    }
    guard let model = contextualEmbedding else { return ["ok": false, "error": "embedding unavailable"] }
    let texts = (args["texts"] as? [String]) ?? []
    var vectors: [[Double]] = []
    for text in texts {
        var sum = [Double](repeating: 0, count: model.dimension)
        var count = 0
        let clipped = String(text.prefix(2000))
        let result = try model.embeddingResult(for: clipped.isEmpty ? " " : clipped, language: .english)
        result.enumerateTokenVectors(in: clipped.startIndex..<clipped.endIndex) { vector, _ in
            for i in 0..<min(vector.count, sum.count) { sum[i] += vector[i] }
            count += 1
            return true
        }
        let mean = sum.map { $0 / Double(max(count, 1)) }
        let norm = max(sqrt(mean.reduce(0) { $0 + $1 * $1 }), 1e-9)
        vectors.append(mean.map { $0 / norm })
    }
    return ["ok": true, "vectors": vectors, "dims": model.dimension, "ms": millis(since: start)]
}

// MARK: - the Neural Engine text encoder (symbio_ane/build_text_encoder.py)

/// MiniLM converted to an fp16 Core ML program with fixed 64-token inputs,
/// loaded with compute units .cpuAndNeuralEngine. It replaced
/// NLContextualEmbedding for the decision model because that one runs on the
/// CPU (measured with macmon: 0 W of Neural Engine while embedding).
nonisolated(unsafe) var textEncoder: MLModel? = nil
nonisolated(unsafe) var textEncoderPath: String = ""

func loadTextEncoder(_ directory: String) throws -> MLModel {
    if let model = textEncoder, textEncoderPath == directory { return model }
    let package = URL(fileURLWithPath: directory).appendingPathComponent("encoder.mlpackage")
    let compiled = URL(fileURLWithPath: directory).appendingPathComponent("encoder.mlmodelc")
    let fm = FileManager.default
    let stale = (try? package.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
        .flatMap { built in
            (try? compiled.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                .map { $0 < built }
        } ?? true
    if !fm.fileExists(atPath: compiled.path) || stale {
        let temporary = try MLModel.compileModel(at: package)
        try? fm.removeItem(at: compiled)
        try fm.moveItem(at: temporary, to: compiled)
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    textEncoder = model
    textEncoderPath = directory
    return model
}

func encode(_ args: [String: Any]) throws -> [String: Any] {
    let start = DispatchTime.now()
    let model = try loadTextEncoder(args["model"] as? String ?? "")
    let ids = (args["ids"] as? [[Int]]) ?? []
    let masks = (args["mask"] as? [[Int]]) ?? []
    var vectors: [[Double]] = []
    for (row, mask) in zip(ids, masks) {
        let idArray = try MLMultiArray(shape: [1, NSNumber(value: row.count)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: mask.count)], dataType: .int32)
        for (i, value) in row.enumerated() { idArray[i] = NSNumber(value: value) }
        for (i, value) in mask.enumerated() { maskArray[i] = NSNumber(value: value) }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray)])
        let output = try model.prediction(from: input)
        guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
            return ["ok": false, "error": "the encoder returned no embedding"]
        }
        vectors.append((0..<embedding.count).map { embedding[$0].doubleValue })
    }
    return ["ok": true, "vectors": vectors, "dims": vectors.first?.count ?? 0,
            "ms": millis(since: start)]
}

// MARK: - decisions (Apple Intelligence's on-device model)

func modelStatus() -> [String: Any] {
    #if canImport(FoundationModels)
    let availability = SystemLanguageModel.default.availability
    switch availability {
    case .available:
        return ["available": true]
    case .unavailable(let reason):
        return ["available": false, "reason": String(describing: reason)]
    @unknown default:
        return ["available": false, "reason": "unknown"]
    }
    #else
    return ["available": false, "reason": "FoundationModels not in this SDK"]
    #endif
}

#if canImport(FoundationModels)
/// The routing question as a schema the model must fill in, not prose to
/// parse: guided generation constrains the tokens to this shape.
func decisionSchema() throws -> GenerationSchema {
    let route = DynamicGenerationSchema(
        name: "route",
        description: "What should handle this message",
        anyOf: ["chat", "tool", "search", "code", "vision", "delegate"])
    let schema = DynamicGenerationSchema(
        name: "Decision",
        description: "How a local assistant should handle the user's latest message",
        properties: [
            .init(name: "think", description: "Does answering well need step-by-step reasoning first?",
                  schema: DynamicGenerationSchema(type: Bool.self)),
            .init(name: "route", schema: route),
            .init(name: "needs_fresh_facts",
                  description: "Does the answer depend on current or external information?",
                  schema: DynamicGenerationSchema(type: Bool.self)),
            .init(name: "reason", description: "Under 12 words",
                  schema: DynamicGenerationSchema(type: String.self)),
        ])
    return try GenerationSchema(root: schema, dependencies: [])
}

nonisolated(unsafe) var readySession: LanguageModelSession? = nil

func newDecisionSession() -> LanguageModelSession {
    // Measured: with a one-line brief the model said think=true for almost
    // everything ("haha that's funny" — "check for sarcasm"), 17/31 held out.
    // The criteria and worked examples below are what it needs; none of the
    // examples is in the held-out set it is graded on.
    LanguageModelSession(instructions: """
        You route messages for a local assistant that runs a large language \
        model. Decide only; never answer the message.

        think = whether the big model must reason step by step BEFORE it \
        answers. It is slow, so it is false unless the answer needs working out.
        think = false: greetings, small talk, thanks, jokes, opinions, quick \
        facts and definitions, one-step arithmetic, anything that is only a \
        lookup of current information (weather, news, prices, scores: those \
        need a search, not thinking), reading or describing the screen, simple \
        commands like opening an app.
        think = true: writing, fixing or explaining code and errors; planning; \
        comparing options or making a decision with trade-offs; multi-step \
        maths or estimates; designing or outlining something.

        Examples:
        "hey there" -> think false, route chat
        "what's 7 times 8?" -> think false, route chat
        "who invented the telephone?" -> think false, route chat
        "weather in perth this weekend" -> think false, route search
        "what does this popup say?" -> think false, route vision
        "open the notes app" -> think false, route tool
        "this function throws IndexError, fix it" -> think true, route code
        "rent or buy in brisbane, which makes more sense for me?" -> think true, route chat
        "how many litres of paint for a 4x5 metre room with 2.4 m walls?" -> think true, route chat
        """)
}

func decide(_ args: [String: Any]) async throws -> [String: Any] {
    let start = DispatchTime.now()
    guard SystemLanguageModel.default.isAvailable else {
        return ["ok": false, "status": modelStatus(), "ms": millis(since: start)]
    }
    let message = args["message"] as? String ?? ""
    // A fresh session per decision (a shared one would grow a transcript of
    // every message it ever routed), taken already warm: the next one is
    // prewarmed as soon as this one is handed out.
    let session = readySession ?? newDecisionSession()
    readySession = nil
    let response = try await session.respond(
        to: "Message: \(message)", schema: try decisionSchema(),
        options: GenerationOptions(sampling: .greedy))
    let content = response.content
    let next = newDecisionSession()
    next.prewarm()
    readySession = next
    return ["ok": true,
            "think": (try? content.value(Bool.self, forProperty: "think")) ?? true,
            "route": (try? content.value(String.self, forProperty: "route")) ?? "chat",
            "needs_fresh_facts": (try? content.value(Bool.self, forProperty: "needs_fresh_facts")) ?? false,
            "reason": (try? content.value(String.self, forProperty: "reason")) ?? "",
            "ms": millis(since: start)]
}
#endif

// MARK: - loop

func handle(_ line: String) async {
    guard let data = line.data(using: .utf8),
          let args = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        reply(["ok": false, "error": "not a JSON object"])
        return
    }
    let op = args["op"] as? String ?? ""
    do {
        switch op {
        case "status":
            reply(["ok": true, "vision": true, "ocr": true, "decide": modelStatus(),
                   "pid": Int(ProcessInfo.processInfo.processIdentifier)])
        case "ocr":
            reply(try ocr(args))
        case "vision":
            reply(try vision(args))
        case "embed":
            reply(try await embed(args))
        case "encode":
            reply(try encode(args))
        case "decide":
            #if canImport(FoundationModels)
            reply(try await decide(args))
            #else
            reply(["ok": false, "status": modelStatus()])
            #endif
        default:
            reply(["ok": false, "error": "unknown op \(op)"])
        }
    } catch {
        reply(["ok": false, "op": op, "error": error.localizedDescription])
    }
}

@main
struct SymbioANE {
    static func main() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        while let line = readLine(strippingNewline: true) {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            await handle(line)
        }
    }
}
