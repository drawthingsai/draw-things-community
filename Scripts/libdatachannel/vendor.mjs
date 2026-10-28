// Keeps Vendors/libdatachannel in step with upstream; Vendors/libdatachannel/UPSTREAM.md has the pins.
//
//   node Scripts/libdatachannel/vendor.mjs watch [--report FILE]
//     Reports upstream releases, advisories and fixes that reviewed.json doesn't cover yet, and
//     exits 1 if there are any.
//   node Scripts/libdatachannel/vendor.mjs verify
//     Checks that the vendored code is upstream's at the pinned commits plus patches/.
//   node Scripts/libdatachannel/vendor.mjs update --libdatachannel vX.Y.Z [--mbedtls 3.6.N]
//     Replaces the vendored code with those releases, the libraries libdatachannel pins included,
//     reapplies patches/ and updates the pins.
import { execFileSync } from "node:child_process";
import {
  cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import reviewedItems from "./reviewed.json" with { type: "json" };

const VENDOR = resolve(dirname(fileURLToPath(import.meta.url)), "../../Vendors/libdatachannel");

// What's copied from each upstream, and the repository the watch checks for fixes to our patches.
export const COMPONENTS = {
  libdatachannel: { dirs: ["src", "include"], license: "LICENSE", watch: "paullouisageneau/libdatachannel" },
  libjuice: { dirs: ["src", "include"], license: "LICENSE", watch: "paullouisageneau/libjuice" },
  usrsctp: { dirs: ["usrsctplib"], license: "LICENSE.md", watch: "sctplab/usrsctp" },
  plog: { dirs: ["include"], license: "LICENSE", watch: "SergiusTheBest/plog" },
  mbedtls: { dirs: ["library", "include"], license: "LICENSE", watch: "Mbed-TLS/mbedtls" },
};
// Upstream build files inside the copied directories, left out.
const OMITTED = new Set(["CMakeLists.txt", "Makefile", "Makefile.am", "Makefile.nmake", "meson.build", ".gitignore"]);
// Files in Vendors/libdatachannel that aren't upstream's.
export const LOCAL = [
  "BUILD.bazel", "Config", "UPSTREAM.md", "patches",
  "libdatachannel/include/module.modulemap", "usrsctp/SwiftPM", "mbedtls/SwiftPM",
];
const MBEDTLS_3_6_SUPPORT_ENDS = Date.UTC(2027, 2, 31);
const MBEDTLS_ADVISORIES = "https://mbed-tls.readthedocs.io/en/latest/security-advisories/";
const MOZILLA_BUG_QUERIES = [
  // Firefox tags usrsctp security bugs, though not all of them.
  "status_whiteboard=%5Busrsctp-sec%5D&status_whiteboard_type=substring",
  "short_desc=sctp&short_desc_type=substring&keywords=sec-critical%20sec-high%20sec-moderate%20sec-low"
    + "&keywords_type=anywords",
].map((query) => `https://bugzilla.mozilla.org/rest/bug?${query}&include_fields=id,summary&limit=1000`);

/** The pins and the patches, in the order they apply, from UPSTREAM.md. */
export function parseUpstream(markdown) {
  const pins = {};
  for (const [, name, url, version, commit] of markdown.matchAll(
    /^\| `(\w+)` \| (https:\/\/github\.com\/[\w.-]+\/[\w.-]+)[^|]*\| ([^|]*)\| `([0-9a-f]{40})` \|/gm,
  )) {
    pins[name] = { url, version: version.trim().replace(/ \(LTS\)$/, ""), commit };
  }
  for (const name of Object.keys(COMPONENTS)) {
    if (!pins[name]) throw new Error(`UPSTREAM.md has no pin for ${name}.`);
  }
  const usrsctpBase = markdown.match(/sctplab `([0-9a-f]{40})`/)?.[1];
  if (!usrsctpBase) throw new Error("UPSTREAM.md doesn't name usrsctp's sctplab commit.");
  const patches = [...markdown.matchAll(/^\| `([\w.-]+\.patch)` \|/gm)].map((match) => match[1]);
  if (!patches.length) throw new Error("UPSTREAM.md lists no patches.");
  return { pins, usrsctpBase, patches };
}

/** The files a patch changes, as paths in Vendors/libdatachannel. */
export function patchedFiles(patch) {
  return [...patch.matchAll(/^\+\+\+ b\/(\S+)/gm)].map((match) => match[1]);
}

/** Whether a patch "applies" to the files in `directory`, is already there ("present"), or neither. */
export function patchStatus(patchFile, directory) {
  const dryRun = (reverse) => {
    try {
      execFileSync("patch", ["-p1", "-s", "-f", "-F", "0", "--dry-run", ...(reverse ? ["-R"] : []),
        "-d", directory, "-i", patchFile], { stdio: "ignore" });
      return true;
    } catch {
      return false;
    }
  };
  if (dryRun(true)) return "present";
  return dryRun(false) ? "applies" : "neither";
}

/**
 * What upstream has that `reviewed` doesn't cover: releases newer than the pins, advisories, Firefox's
 * usrsctp fixes and Mozilla's SCTP security bugs, and upstream fixes for our own patches. A source
 * that can't be checked is a finding too.
 */
export async function watch({ upstream, reviewed, sources, now = Date.now() }) {
  const { github, getJSON, getText, ownPatchStatus } = sources;
  const { pins } = upstream;
  const findings = [];
  const check = async (what, run) => {
    try {
      findings.push(...await run());
    } catch (error) {
      findings.push(`Couldn't check ${what}: ${error.message}`);
    }
  };

  await check("libdatachannel's releases", async () => {
    const tag = (await github("/repos/paullouisageneau/libdatachannel/releases/latest")).tag_name;
    return tag === pins.libdatachannel.version ? [] : [
      `libdatachannel ${tag} is out; the vendored code is ${pins.libdatachannel.version}. `
        + `See https://github.com/paullouisageneau/libdatachannel/releases/tag/${tag}`,
    ];
  });
  await check("libjuice's releases", async () => {
    const tag = (await github("/repos/paullouisageneau/libjuice/releases/latest")).tag_name;
    return tag.replace(/^v/, "") === pins.libjuice.version ? [] : [
      `libjuice ${tag} is out; libdatachannel pins ${pins.libjuice.version}. Check its fixes: `
        + `https://github.com/paullouisageneau/libjuice/releases/tag/${tag}`,
    ];
  });
  await check("Mbed TLS's releases", async () => {
    const releases = await github("/repos/Mbed-TLS/mbedtls/releases?per_page=50");
    const patches = releases.map((release) => release.tag_name.match(/^mbedtls-3\.6\.(\d+)$/)?.[1])
      .filter(Boolean).map(Number);
    if (!patches.length) throw new Error("no 3.6 release in the list");
    const newest = Math.max(...patches);
    const findings = newest > Number(pins.mbedtls.version.split(".")[2]) ? [
      `Mbed TLS 3.6.${newest} is out; the vendored code is ${pins.mbedtls.version}. `
        + `See https://github.com/Mbed-TLS/mbedtls/releases/tag/mbedtls-3.6.${newest}`,
    ] : [];
    if (now > MBEDTLS_3_6_SUPPORT_ENDS - 120 * 24 * 60 * 60 * 1000) {
      findings.push("Mbed TLS 3.6 support ends on 2027-03-31: move to Mbed TLS 4 with a libdatachannel "
        + "release that supports it.");
    }
    return findings;
  });
  await check("Mbed TLS's advisories", async () => {
    const page = await getText(MBEDTLS_ADVISORIES);
    const advisories = new Map();
    for (const [, slug, month] of page.matchAll(/href="(?:[^"]*\/)?(mbedtls-security-advisory-(\d{4}-\d{2})-[\w-]+)\/?"/g)) {
      advisories.set(slug, month);
    }
    if (!advisories.size) throw new Error("no advisories on the page");
    return [...advisories].filter(([, month]) => month > reviewed.mbedtlsAdvisoriesThrough)
      .map(([slug]) => `New Mbed TLS advisory: ${MBEDTLS_ADVISORIES}${slug}/`);
  });
  await check("Firefox's usrsctp patches", async () => {
    const files = await github("/repos/mozilla-firefox/firefox/contents/netwerk/sctp/src?ref=main");
    if (!Array.isArray(files) || !files.length) throw new Error("netwerk/sctp/src is empty or gone");
    return files.map((file) => file.name).filter((name) => name.endsWith(".patch"))
      .filter((name) => !(name in reviewed.firefoxSctpPatches))
      .map((name) => `New Firefox usrsctp patch: https://github.com/mozilla-firefox/firefox/blob/main/netwerk/sctp/src/${name}`);
  });
  await check("Mozilla's SCTP security bugs", async () => {
    const bugs = new Map();
    for (const url of MOZILLA_BUG_QUERIES) {
      const response = await getJSON(url);
      if (!Array.isArray(response.bugs)) throw new Error("Bugzilla's response has no bug list");
      for (const bug of response.bugs) bugs.set(String(bug.id), bug.summary);
    }
    return [...bugs].filter(([id]) => !(id in reviewed.mozillaBugs))
      .map(([id, summary]) => `New Mozilla SCTP security bug ${id}: ${summary} `
        + `(https://bugzilla.mozilla.org/show_bug.cgi?id=${id})`);
  });
  await check("OSV", async () => {
    const findings = [];
    const commits = [...Object.entries(pins).map(([name, pin]) => [name, pin.commit]),
      ["usrsctp (sctplab base)", upstream.usrsctpBase]];
    for (const [name, commit] of commits) {
      const { vulns = [] } = await getJSON("https://api.osv.dev/v1/query", { commit });
      for (const { id } of vulns) {
        if (!(id in reviewed.advisories)) findings.push(`OSV lists ${id} for ${name}: https://osv.dev/vulnerability/${id}`);
      }
    }
    return findings;
  });
  await check("GitHub's advisories", async () => {
    const findings = [];
    for (const { watch: repo } of Object.values(COMPONENTS)) {
      for (const advisory of await github(`/repos/${repo}/security-advisories?state=published&per_page=100`)) {
        if (!(advisory.ghsa_id in reviewed.advisories)) {
          findings.push(`New advisory for ${repo}: ${advisory.summary} (${advisory.html_url})`);
        }
      }
    }
    return findings;
  });
  for (const name of upstream.patches) {
    await check(`whether upstream fixed patches/${name}`, async () => {
      const { status, repo } = await ownPatchStatus(name);
      if (status === "present") return [`${repo} now has the fix in patches/${name}; drop the patch at the next update.`];
      if (status === "neither") return [`patches/${name} no longer applies to ${repo}'s latest code; port it at the next update.`];
      return [];
    });
  }
  return findings;
}

/** Fetches the files a patch changes from upstream's default branch and checks the patch against them. */
async function upstreamPatchStatus(name, github) {
  const patchFile = join(VENDOR, "patches", name);
  const work = mkdtempSync(join(tmpdir(), "libdatachannel-watch-"));
  try {
    const repos = new Set();
    for (const path of patchedFiles(readFileSync(patchFile, "utf8"))) {
      const [component, ...rest] = path.split("/");
      const repo = COMPONENTS[component].watch;
      repos.add(repo);
      mkdirSync(dirname(join(work, path)), { recursive: true });
      writeFileSync(join(work, path), await github(`/repos/${repo}/contents/${rest.join("/")}`, { raw: true }));
    }
    return { status: patchStatus(patchFile, work), repo: [...repos].join(" and ") };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

const liveSources = {
  async github(path, { raw = false } = {}) {
    const token = process.env.GITHUB_TOKEN;
    const response = await fetch(`https://api.github.com${path}`, {
      headers: {
        accept: raw ? "application/vnd.github.raw" : "application/vnd.github+json",
        "user-agent": "draw-things-libdatachannel-watch",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
    });
    if (!response.ok) throw new Error(`GitHub ${path}: HTTP ${response.status}`);
    return raw ? response.text() : response.json();
  },
  async getJSON(url, body) {
    const response = await fetch(url, body
      ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
      : {});
    if (!response.ok) throw new Error(`${new URL(url).host}: HTTP ${response.status}`);
    return response.json();
  },
  async getText(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${new URL(url).host}: HTTP ${response.status}`);
    return response.text();
  },
  ownPatchStatus: (name) => upstreamPatchStatus(name, liveSources.github),
};

export function formatReport(findings) {
  if (!findings.length) return "Nothing new upstream for Vendors/libdatachannel.\n";
  return [
    "# libdatachannel upstream needs attention", "",
    ...findings.map((finding) => `- ${finding}`), "",
    "For each item, update the vendored code (`node Scripts/libdatachannel/vendor.mjs update`), add a patch "
      + "to `Vendors/libdatachannel/patches`, or record why no action is needed in "
      + "`Scripts/libdatachannel/reviewed.json`. `Vendors/libdatachannel/UPSTREAM.md` has the details.",
  ].join("\n") + "\n";
}

function git(directory, ...args) {
  return execFileSync("git", ["-C", directory, ...args], { encoding: "utf8" }).trim();
}

/** A shallow checkout of `ref` from `url` in `directory`. */
export function gitCheckout(url, ref, directory) {
  execFileSync("git", ["init", "-q", directory]);
  git(directory, "fetch", "-q", "--depth", "1", url, ref);
  git(directory, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD");
  return directory;
}

/** Copies each component's upstream subset at its pin into `into`, laid out like Vendors/libdatachannel. */
export function assemble(pins, into, work, checkout = gitCheckout) {
  for (const [name, { dirs, license }] of Object.entries(COMPONENTS)) {
    const source = checkout(pins[name].url, pins[name].commit, join(work, name));
    for (const dir of dirs) {
      cpSync(join(source, dir), join(into, name, dir), {
        recursive: true, filter: (path) => !OMITTED.has(basename(path)),
      });
    }
    cpSync(join(source, license), join(into, name, license));
  }
}

function listFiles(root) {
  return readdirSync(root, { recursive: true })
    .filter((path) => basename(path) !== ".DS_Store" && statSync(join(root, path)).isFile());
}

/** How `actual` differs from `expected`, leaving out `local` paths. */
export function compareTrees(expected, actual, local = []) {
  const isLocal = (path) => local.some((entry) => path === entry || path.startsWith(`${entry}/`));
  const want = new Set(listFiles(expected).filter((path) => !isLocal(path)));
  const have = new Set(listFiles(actual).filter((path) => !isLocal(path)));
  const differences = [];
  for (const path of have) {
    if (!want.has(path)) {
      differences.push(`${path}: not upstream's, and not one of the local files`);
    } else if (!readFileSync(join(expected, path)).equals(readFileSync(join(actual, path)))) {
      differences.push(`${path}: differs from upstream plus patches/`);
    }
  }
  for (const path of want) if (!have.has(path)) differences.push(`${path}: missing`);
  return differences.sort();
}

const APPLY = ["-p1", "-s", "-f", "-F", "0"];

/** Differences between Vendors/libdatachannel and upstream at the pins plus patches/. */
export function verify(upstream, vendor = VENDOR, checkout = gitCheckout) {
  const work = mkdtempSync(join(tmpdir(), "libdatachannel-verify-"));
  try {
    const expected = join(work, "expected");
    assemble(upstream.pins, expected, work, checkout);
    for (const name of upstream.patches) {
      execFileSync("patch", [...APPLY, "-d", expected, "-i", join(vendor, "patches", name)], { stdio: "ignore" });
    }
    return compareTrees(expected, vendor, LOCAL);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

/** UPSTREAM.md with each component's version and commit replaced by `pins`. */
export function updatePins(markdown, oldPins, pins) {
  return markdown.replace(
    /^(\| `(\w+)` \| [^|]*\| )([^|]*)(\| `)([0-9a-f]{40})(` \|)/gm,
    (row, start, name, version, open, commit, close) => !pins[name] ? row
      : start + version.replace(oldPins[name].version, pins[name].version) + open + pins[name].commit + close,
  );
}

/** Replaces the vendored code with a libdatachannel release, and optionally an Mbed TLS 3.6 release. */
export function update(upstream, releases, vendor = VENDOR) {
  const work = mkdtempSync(join(tmpdir(), "libdatachannel-update-"));
  try {
    const pins = structuredClone(upstream.pins);
    const libdatachannel = gitCheckout(pins.libdatachannel.url, `refs/tags/${releases.libdatachannel}`,
      join(work, "libdatachannel"));
    pins.libdatachannel.version = releases.libdatachannel;
    pins.libdatachannel.commit = git(libdatachannel, "rev-parse", "HEAD");
    const modules = readFileSync(join(libdatachannel, ".gitmodules"), "utf8");
    for (const name of ["libjuice", "usrsctp", "plog"]) {
      const url = modules.match(new RegExp(`path = deps/${name}\\s+url = (\\S+)`))?.[1];
      if (!url) throw new Error(`libdatachannel ${releases.libdatachannel} has no deps/${name} submodule.`);
      pins[name].url = url.replace(/\.git$/, "");
      pins[name].commit = git(libdatachannel, "rev-parse", `HEAD:deps/${name}`);
    }
    if (releases.mbedtls) {
      const mbedtls = gitCheckout(pins.mbedtls.url, `refs/tags/mbedtls-${releases.mbedtls}`, join(work, "mbedtls"));
      pins.mbedtls.version = releases.mbedtls;
      pins.mbedtls.commit = git(mbedtls, "rev-parse", "HEAD");
    }
    const expected = join(work, "expected");
    assemble(pins, expected, work);
    const juiceVersion = readFileSync(join(work, "libjuice", "CMakeLists.txt"), "utf8")
      .match(/project\s*\(\s*libjuice[^)]*VERSION\s+([\d.]+)/)?.[1];
    if (juiceVersion) pins.libjuice.version = juiceVersion;

    const upstreamHasIt = [];
    for (const name of upstream.patches) {
      const patchFile = join(vendor, "patches", name);
      const status = patchStatus(patchFile, expected);
      if (status === "present") {
        upstreamHasIt.push(name);
      } else if (status === "neither") {
        throw new Error(`patches/${name} doesn't apply to the new code; port it by hand.`);
      } else {
        execFileSync("patch", [...APPLY, "-d", expected, "-i", patchFile], { stdio: "ignore" });
      }
    }

    // The local files inside the components survive the copy.
    const kept = join(work, "kept");
    const insideComponents = LOCAL.filter((path) => path.includes("/"));
    for (const path of insideComponents) {
      if (existsSync(join(vendor, path))) cpSync(join(vendor, path), join(kept, path), { recursive: true });
    }
    for (const name of Object.keys(COMPONENTS)) {
      rmSync(join(vendor, name), { recursive: true, force: true });
      cpSync(join(expected, name), join(vendor, name), { recursive: true });
    }
    for (const path of insideComponents) {
      if (existsSync(join(kept, path))) cpSync(join(kept, path), join(vendor, path), { recursive: true });
    }
    const upstreamFile = join(vendor, "UPSTREAM.md");
    writeFileSync(upstreamFile, updatePins(readFileSync(upstreamFile, "utf8"), upstream.pins, pins));
    return { pins, upstreamHasIt };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

function option(args, name) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

async function main([command, ...args]) {
  const upstream = parseUpstream(readFileSync(join(VENDOR, "UPSTREAM.md"), "utf8"));
  if (command === "watch") {
    const findings = await watch({ upstream, reviewed: reviewedItems, sources: liveSources });
    const report = formatReport(findings);
    const reportFile = option(args, "--report");
    if (reportFile) writeFileSync(reportFile, report);
    process.stdout.write(report);
    process.exitCode = findings.length ? 1 : 0;
  } else if (command === "verify") {
    const differences = verify(upstream);
    if (differences.length) {
      console.error(`Vendors/libdatachannel isn't upstream's code at the pins plus patches/:\n${differences.join("\n")}`);
      process.exitCode = 1;
    } else {
      console.log("Vendors/libdatachannel is upstream's code at the pins plus patches/.");
    }
  } else if (command === "update" && option(args, "--libdatachannel")) {
    const { pins, upstreamHasIt } = update(upstream, {
      libdatachannel: option(args, "--libdatachannel"), mbedtls: option(args, "--mbedtls"),
    });
    for (const [name, pin] of Object.entries(pins)) console.log(`${name}: ${pin.version} ${pin.commit} (${pin.url})`);
    for (const name of upstreamHasIt) {
      console.log(`Upstream has patches/${name}: delete it and its row in UPSTREAM.md.`);
    }
    console.log("Next: check upstream's CMakeLists.txt files for new sources or definitions, update "
      + "UPSTREAM.md's notes, then run the validation in UPSTREAM.md and `vendor.mjs verify`.");
  } else {
    console.error("usage: vendor.mjs watch [--report FILE] | verify | "
      + "update --libdatachannel vX.Y.Z [--mbedtls 3.6.N]");
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) await main(process.argv.slice(2));
