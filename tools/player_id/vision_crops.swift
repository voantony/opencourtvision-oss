// vision_crops: Apple Vision person instance mask + 2D body pose for single-person crops (no faces used or stored).
// Usage: vision_crops <list.txt> <out.jsonl>   (list: one crop JPEG path per line; masks written next to each crop as <crop>.mask.u8)
// Output JSON lines: {"path":..,"kp":[[x,y,c]*17] (COCO-17, crop-normalised, top-left origin),"mw":w,"mh":h,"mask":0|1,"ms_mask":..,"ms_pose":..}
// The mask is the instance under the crop centre, as raw uint8 (0/255) of mw x mh. Build: swiftc -O vision_crops.swift -o vision_crops
import Foundation
import Vision
import CoreImage
import ImageIO

let a = CommandLine.arguments
guard a.count >= 3 else { FileHandle.standardError.write("usage: vision_crops list.txt out.jsonl\n".data(using: .utf8)!); exit(2) }
let paths = (try! String(contentsOfFile: a[1], encoding: .utf8)).split(separator: "\n").map(String.init).filter { !$0.isEmpty }
FileManager.default.createFile(atPath: a[2], contents: nil)
let fh = FileHandle(forWritingAtPath: a[2])!
let order: [VNHumanBodyPoseObservation.JointName] = [.nose, .leftEye, .rightEye, .leftEar, .rightEar, .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
                                                     .leftWrist, .rightWrist, .leftHip, .rightHip, .leftKnee, .rightKnee, .leftAnkle, .rightAnkle]
for p in paths {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: p) as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
    let h = VNImageRequestHandler(cgImage: img, options: [:])
    var rec: [String: Any] = ["path": p, "mask": 0]
    // mask
    let t0 = Date()
    let mr = VNGeneratePersonInstanceMaskRequest()
    if (try? h.perform([mr])) != nil, let obs = mr.results?.first {
        let inst = obs.instanceMask
        CVPixelBufferLockBaseAddress(inst, .readOnly)
        let w = CVPixelBufferGetWidth(inst), hh = CVPixelBufferGetHeight(inst), rb = CVPixelBufferGetBytesPerRow(inst)
        let base = CVPixelBufferGetBaseAddress(inst)!.assumingMemoryBound(to: UInt8.self)
        var counts = [Int](repeating: 0, count: 256)
        for y in Int(Double(hh) * 0.3)..<Int(Double(hh) * 0.7) { for x in Int(Double(w) * 0.35)..<Int(Double(w) * 0.65) { counts[Int(base[y * rb + x])] += 1 } }
        counts[0] = 0
        if let best = counts.indices.max(by: { counts[$0] < counts[$1] }), counts[best] > 0 {
            var out = [UInt8](repeating: 0, count: w * hh)
            for y in 0..<hh { for x in 0..<w { if base[y * rb + x] == UInt8(best) { out[y * w + x] = 255 } } }
            FileManager.default.createFile(atPath: p + ".mask.u8", contents: Data(out))
            rec["mask"] = 1; rec["mw"] = w; rec["mh"] = hh
        }
        CVPixelBufferUnlockBaseAddress(inst, .readOnly)
    }
    rec["ms_mask"] = Date().timeIntervalSince(t0) * 1000
    // pose: the skeleton nearest the crop centre
    let t1 = Date()
    let bp = VNDetectHumanBodyPoseRequest()
    try? h.perform([bp])
    var best: [[Double]]? = nil; var bd = 1e9
    for o in bp.results ?? [] {
        var kp: [[Double]] = []; var xs: [Double] = [], ys: [Double] = []
        for j in order {
            if let q = try? o.recognizedPoint(j), q.confidence > 0.1 { let x = Double(q.location.x), y = 1 - Double(q.location.y); kp.append([x, y, Double(q.confidence)]); xs.append(x); ys.append(y) }
            else { kp.append([0, 0, 0]) }
        }
        guard xs.count >= 4 else { continue }
        let cx = xs.reduce(0, +) / Double(xs.count), cy = ys.reduce(0, +) / Double(ys.count)
        let d = (cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5)
        if d < bd { bd = d; best = kp }
    }
    rec["ms_pose"] = Date().timeIntervalSince(t1) * 1000
    if let b = best { rec["kp"] = b }
    if let d = try? JSONSerialization.data(withJSONObject: rec) { fh.write(d); fh.write("\n".data(using: .utf8)!) }
}
fh.closeFile()
