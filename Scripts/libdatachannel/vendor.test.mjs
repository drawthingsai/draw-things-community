import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import reviewed from "./reviewed.json" with { type: "json" };
import {
  compareTrees, formatReport, parseUpstream, patchedFiles, patchStatus, updatePins, watch,
} from "./vendor.mjs";

const upstreamFile = new URL("../../Vendors/libdatachannel/UPSTREAM.md", import.meta.url);
const markdown = readFileSync(upstreamFile, "utf8");
const upstream = parseUpstream(markdown);
const today = Date.UTC(2026, 9, 28);

// Upstream as it was when reviewed.json was last updated; tests change one thing at a time.
function sources(changes = {}) {
  const state = {
    releases: { "paullouisageneau/libdatachannel": "v0.24.6", "paullouisageneau/libjuice": "v1.7.4" },
    mbedtlsTags: ["mbedtls-4.2.0", "mbedtls-4.1.1", "mbedtls-3.6.7", "mbedtls-3.6.6"],
    advisoriesPage: '<a href="mbedtls-security-advisory-2026-07-zero-length-ecc-public-key-oob-read/">x</a>'
      + '<a href="../mbedtls-security-advisory-2025-10-earlier/">y</a>',
    firefoxFiles: [...Object.keys(reviewed.firefoxSctpPatches), "moz.yaml"],
    bugs: Object.keys(reviewed.mozillaBugs).map((id) => ({ id: Number(id), summary: "reviewed" })),
    osv: { [upstream.usrsctpBase]: ["OSV-2021-343"] },
    advisories: {},
    patchStatus: "applies",
    ...changes,
  };
  return {
    async github(path) {
      const release = path.match(/^\/repos\/(.+)\/releases\/latest$/);
      if (release) return { tag_name: state.releases[release[1]] };
      if (path.startsWith("/repos/Mbed-TLS/mbedtls/releases")) return state.mbedtlsTags.map((tag_name) => ({ tag_name }));
      if (path.startsWith("/repos/mozilla-firefox/firefox/contents/")) return state.firefoxFiles.map((name) => ({ name }));
      const advisories = path.match(/^\/repos\/(.+)\/security-advisories/);
      if (advisories) return state.advisories[advisories[1]] ?? [];
      throw new Error(`unexpected request ${path}`);
    },
    async getJSON(url, body) {
      if (url.includes("osv.dev")) return { vulns: (state.osv[body.commit] ?? []).map((id) => ({ id })) };
      if (state.bugs instanceof Error) throw state.bugs;
      return { bugs: state.bugs };
    },
    async getText() {
      return state.advisoriesPage;
    },
    async ownPatchStatus() {
      return { status: state.patchStatus, repo: "sctplab/usrsctp" };
    },
  };
}
const findings = (changes, now = today) => watch({ upstream, reviewed, sources: sources(changes), now });

test("reads the pins, usrsctp's sctplab base and the patches in order from UPSTREAM.md", () => {
  assert.deepEqual(Object.keys(upstream.pins), ["libdatachannel", "libjuice", "usrsctp", "plog", "mbedtls"]);
  assert.equal(upstream.pins.libdatachannel.version, "v0.24.6");
  assert.equal(upstream.pins.libjuice.version, "1.7.4");
  assert.equal(upstream.pins.mbedtls.version, "3.6.7");
  assert.equal(upstream.pins.usrsctp.url, "https://github.com/paullouisageneau/usrsctp");
  assert.match(upstream.usrsctpBase, /^[0-9a-f]{40}$/);
  assert.deepEqual(upstream.patches, [
    "libdatachannel-verify-certificate.patch", "usrsctp-tsn-gap-check.patch", "usrsctp-flush-before-process-init.patch",
  ]);
});
test("fails closed when UPSTREAM.md loses a pin or its patches", () => {
  assert.throws(() => parseUpstream(markdown.replace(/^\| `plog`.*$/m, "")), /plog/);
  assert.throws(() => parseUpstream(markdown.replaceAll(".patch` |", "` |")), /no patches/);
});
test("every patch names the vendored files it changes", () => {
  for (const name of upstream.patches) {
    const files = patchedFiles(readFileSync(new URL(`../../Vendors/libdatachannel/patches/${name}`, import.meta.url), "utf8"));
    assert.ok(files.length && files.every((path) => /^(libdatachannel|usrsctp)\//.test(path)), name);
  }
});
test("reports nothing when upstream has nothing reviewed.json doesn't cover", async () => {
  assert.deepEqual(await findings(), []);
});
test("reports new libdatachannel, libjuice and Mbed TLS 3.6 releases", async () => {
  const report = (await findings({
    releases: { "paullouisageneau/libdatachannel": "v0.24.7", "paullouisageneau/libjuice": "v1.7.5" },
    mbedtlsTags: ["mbedtls-4.3.0", "mbedtls-3.6.8", "mbedtls-3.6.7"],
  })).join("\n");
  assert.match(report, /libdatachannel v0\.24\.7 is out/);
  assert.match(report, /libjuice v1\.7\.5 is out/);
  assert.match(report, /Mbed TLS 3\.6\.8 is out/);
});
test("reports advisories, fixes and bugs that aren't reviewed yet", async () => {
  const report = (await findings({
    advisoriesPage: '<a href="mbedtls-security-advisory-2026-11-dtls-overflow/">new</a>',
    firefoxFiles: ["tsn-gap-check.patch", "new-fix.patch"],
    bugs: [{ id: 2045773, summary: "reviewed" }, { id: 2099999, summary: "Overflow in sctp_handle_x" }],
    osv: { [upstream.pins.libjuice.commit]: ["OSV-2026-1"] },
    advisories: { "sctplab/usrsctp": [{ ghsa_id: "GHSA-aaaa-bbbb-cccc", summary: "UAF", html_url: "https://x" }] },
  })).join("\n");
  assert.match(report, /mbedtls-security-advisory-2026-11-dtls-overflow/);
  assert.match(report, /new-fix\.patch/);
  assert.match(report, /bug 2099999: Overflow in sctp_handle_x/);
  assert.doesNotMatch(report, /2045773/);
  assert.match(report, /OSV-2026-1 for libjuice/);
  assert.match(report, /New advisory for sctplab\/usrsctp: UAF/);
});
test("reports when upstream has one of our fixes, or a patch no longer applies", async () => {
  assert.equal((await findings({ patchStatus: "present" })).filter((line) => line.includes("drop the patch")).length, 3);
  assert.equal((await findings({ patchStatus: "neither" })).filter((line) => line.includes("port it")).length, 3);
});
test("a source that can't be checked is a finding, not silence", async () => {
  const report = (await findings({ bugs: new Error("HTTP 503"), advisoriesPage: "<html></html>" })).join("\n");
  assert.match(report, /Couldn't check Mozilla's SCTP security bugs: HTTP 503/);
  assert.match(report, /Couldn't check Mbed TLS's advisories: no advisories/);
});
test("warns about the end of Mbed TLS 3.6 support from December 2026", async () => {
  assert.match((await findings({}, Date.UTC(2027, 0, 15))).join("\n"), /Mbed TLS 3\.6 support ends/);
});
test("the report says what to do with each finding", () => {
  assert.match(formatReport(["x"]), /reviewed\.json/);
  assert.match(formatReport([]), /Nothing new/);
});

test("tells a patch that applies from one that's already there", (t) => {
  const work = mkdtempSync(join(tmpdir(), "vendor-test-"));
  t.after(() => rmSync(work, { recursive: true, force: true }));
  mkdirSync(join(work, "tree/lib"), { recursive: true });
  const file = join(work, "tree/lib/a.c");
  writeFileSync(file, "one\ntwo\nthree\nfour\n");
  const patch = join(work, "fix.patch");
  writeFileSync(patch, "--- a/lib/a.c\n+++ b/lib/a.c\n@@ -1,4 +1,5 @@\n one\n two\n+check\n three\n four\n");
  assert.equal(patchStatus(patch, join(work, "tree")), "applies");
  writeFileSync(file, "one\ntwo\ncheck\nthree\nfour\n");
  assert.equal(patchStatus(patch, join(work, "tree")), "present");
  writeFileSync(file, "rewritten\n");
  assert.equal(patchStatus(patch, join(work, "tree")), "neither");
});
test("compares trees file by file, leaving out local files", (t) => {
  const work = mkdtempSync(join(tmpdir(), "vendor-test-"));
  t.after(() => rmSync(work, { recursive: true, force: true }));
  for (const [path, text] of [
    ["expected/a/same.c", "x"], ["expected/a/changed.c", "x"], ["expected/a/missing.c", "x"],
    ["actual/a/same.c", "x"], ["actual/a/changed.c", "y"], ["actual/a/extra.c", "x"], ["actual/BUILD.bazel", "local"],
  ]) {
    mkdirSync(join(work, path, ".."), { recursive: true });
    writeFileSync(join(work, path), text);
  }
  assert.deepEqual(compareTrees(join(work, "expected"), join(work, "actual"), ["BUILD.bazel"]), [
    "a/changed.c: differs from upstream plus patches/",
    "a/extra.c: not upstream's, and not one of the local files",
    "a/missing.c: missing",
  ]);
});
test("updates a pin's version and commit and keeps the rest of its row", () => {
  const pins = structuredClone(upstream.pins);
  pins.mbedtls.version = "3.6.8";
  pins.mbedtls.commit = "a".repeat(40);
  const updated = updatePins(markdown, upstream.pins, pins);
  assert.match(updated, /^\| `mbedtls` \| https:\/\/github\.com\/Mbed-TLS\/mbedtls \| 3\.6\.8 \(LTS\) \| `a{40}` \| Apache-2\.0 \|$/m);
  assert.equal(updated.replace(/^\| `mbedtls`.*$/m, ""), markdown.replace(/^\| `mbedtls`.*$/m, ""));
});
