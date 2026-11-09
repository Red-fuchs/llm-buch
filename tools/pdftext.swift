// pdftext FILE.pdf -> JSON lines {"page":N,"text":"..."} (N = 1-based PDF page). Used by `buch build`.
import Foundation
import PDFKit
guard CommandLine.arguments.count > 1, let doc = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else { FileHandle.standardError.write("open failed\n".data(using: .utf8)!); exit(1) }
for i in 0..<doc.pageCount {
    let t = doc.page(at: i)?.string ?? ""
    let d = try! JSONSerialization.data(withJSONObject: ["page": i + 1, "text": t], options: [.withoutEscapingSlashes])
    print(String(data: d, encoding: .utf8)!)
}
