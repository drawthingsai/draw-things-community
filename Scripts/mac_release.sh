#!/usr/bin/env bash
# Builds, signs and packages one Mac Catalyst app for both the Mac App Store and Developer ID
# distribution. Run it on the release Mac from a maccatalyst checkout, with the patched Bazel at
# ../bazel/bazel-dev, in a login shell (Terminal, or `zsh -lc` over SSH) so Bazel sees the same
# PATH as other builds:
#
#   Scripts/mac_release.sh <localcode|drawthings> [--upload]
#
# The version comes from the app's SupportingFiles/Info.plist. Results go to
# ~/Releases/<App>-<version>/: <App>.pkg for the App Store, and a notarized, stapled
# <App>-<version>-<digest>.zip for direct downloads, plus appcast.xml when the app updates through
# Sparkle and a notarized <App>-<version>-<digest>.dmg when it has a disk image layout. Once both
# are built, --upload publishes the zip, disk image and appcast.xml to the app's release bucket,
# then sends the App Store package to App Store Connect.

set -euo pipefail

usage() {
  echo "Usage: $0 <localcode|drawthings> [--upload]" >&2
  exit 64
}

[[ $# -ge 1 ]] || usage
APP=$1
shift
UPLOAD=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --upload) UPLOAD=1 ;;
    *) usage ;;
  esac
  shift
done

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"
BAZEL=${BAZEL:-../bazel/bazel-dev}

APP_STORE_IDENTITY="Apple Distribution: Draw Things, Inc. (6HP3U6Z7P6)"
DEVELOPER_ID_IDENTITY="Developer ID Application: Liu Liu (6HP3U6Z7P6)"
INSTALLER_IDENTITY="3rd Party Mac Developer Installer: Draw Things, Inc. (6HP3U6Z7P6)"
NOTARY_PROFILE="NOTARIZE_APP_CREDENTIALS"
ASC_API_KEY="U8PV34K96F"
ASC_API_ISSUER="69a6de83-36b3-47e3-e053-5b8c7c11a4d1"
SPARKLE_VERSION="2.10.0"
SPARKLE_SHA256="c2bf58aa8387266ac179357b1415d6f2635f044da8be41042af32425dae6da0c"
R2_ACCOUNT_ID="cd96f610b0bb2657da157aca332052ec"
DMGBUILD_VERSION="1.6.7"
DMGBUILD_REQUIREMENTS="\
dmgbuild==$DMGBUILD_VERSION --hash=sha256:37ee5771c377beb3203d9164aae8046ffed8531c06edf9227f5788b3c599b1bf
ds-store==1.3.3 --hash=sha256:b92a371efbf1b4ccce2a04d1ed13fceacc4736c81ba09cf5aefb74c088160a35
mac-alias==2.2.3 --hash=sha256:7362b521d2132ef92f606a37abfed5fcd849ceb2f28b6f9743e014b02af92f0d"

# EXCLUDED_RESOURCES are Bazel resources that the Mac app has not shipped so far. RELEASE_BUCKET is
# the R2 bucket that serves direct downloads and the app's SUFeedURL.
case $APP in
  localcode)
    SOURCE=Apps/LocalCode
    TARGET=//Apps/LocalCode:LocalCodeMac
    BUILT_NAME=LocalCodeMac
    APP_NAME="Local Code"
    EXECUTABLE=LocalCode
    ARCHIVE_NAME=LocalCode
    BUNDLE_ID=com.liuliu.local-code
    CPUS=arm64
    EXCLUDED_RESOURCES=(AppIcon.icns LICENSE local-code-banner-launch@2x.png Notices)
    RELEASE_BUCKET=local-code-releases
    ;;
  drawthings)
    SOURCE=Apps/DrawThings
    TARGET=//Apps/DrawThings:DrawThingsMac
    BUILT_NAME=DrawThingsMac
    APP_NAME="Draw Things"
    EXECUTABLE=DrawThings
    ARCHIVE_NAME=DrawThings
    BUNDLE_ID=com.liuliu.draw-things
    CPUS=arm64,x86_64
    EXCLUDED_RESOURCES=(AppIcon.icns queue-icon.json)
    RELEASE_BUCKET=""
    ;;
  *) usage ;;
esac
MAC_FILES=$SOURCE/SupportingFiles/Mac
APP_STORE_PROFILE=$SOURCE/ProvisioningFiles/${ARCHIVE_NAME}_macOS_Distribution.provisionprofile
DEVELOPER_ID_PROFILE=$SOURCE/ProvisioningFiles/${ARCHIVE_NAME}_Developer_ID_Distribution.provisionprofile
# Apps with Sparkle.plist update through Sparkle in their direct builds, and apps with
# DiskImage.py (dmgbuild settings) also ship them as a disk image.
SPARKLE_KEYS=$MAC_FILES/Sparkle.plist
DISK_IMAGE_SETTINGS=$MAC_FILES/DiskImage.py

for profile in "$APP_STORE_PROFILE" "$DEVELOPER_ID_PROFILE"; do
  [[ -f $profile ]] || {
    echo "Missing provisioning profile $profile" >&2
    exit 1
  }
done
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  echo "Commit or stash local changes before releasing:" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi
for identity in "$APP_STORE_IDENTITY" "$DEVELOPER_ID_IDENTITY" "$INSTALLER_IDENTITY"; do
  security find-identity -v | grep -qF "\"$identity\"" || {
    echo "Missing signing identity \"$identity\"" >&2
    exit 1
  }
done
# Codesign and notarytool read the login keychain, which is locked in SSH sessions.
security show-keychain-info "$HOME/Library/Keychains/login.keychain-db" >/dev/null 2>&1 ||
  security unlock-keychain "$HOME/Library/Keychains/login.keychain-db"

if [[ -f $SPARKLE_KEYS ]]; then
  SPARKLE_DIR=$HOME/Library/Caches/mac_release/Sparkle-$SPARKLE_VERSION
  if [[ ! -d $SPARKLE_DIR/Sparkle.framework ]]; then
    DOWNLOAD=$(mktemp -d)
    curl -fsSL -o "$DOWNLOAD/Sparkle.tar.xz" \
      "https://github.com/sparkle-project/Sparkle/releases/download/$SPARKLE_VERSION/Sparkle-$SPARKLE_VERSION.tar.xz"
    echo "$SPARKLE_SHA256  $DOWNLOAD/Sparkle.tar.xz" | shasum -a 256 -c - >/dev/null
    mkdir -p "$DOWNLOAD/Sparkle"
    tar -xJf "$DOWNLOAD/Sparkle.tar.xz" -C "$DOWNLOAD/Sparkle"
    mkdir -p "$(dirname "$SPARKLE_DIR")"
    rm -rf "$SPARKLE_DIR"
    mv "$DOWNLOAD/Sparkle" "$SPARKLE_DIR"
    rm -rf "$DOWNLOAD"
  fi
  # An appcast signed with a key the shipped app does not trust breaks updates for everyone.
  KEYCHAIN_KEY=$("$SPARKLE_DIR/bin/generate_keys" -p) || {
    echo "No Sparkle signing key in the login keychain; create one with $SPARKLE_DIR/bin/generate_keys" >&2
    exit 1
  }
  [[ $KEYCHAIN_KEY == "$(plutil -extract SUPublicEDKey raw "$SPARKLE_KEYS")" ]] || {
    echo "SUPublicEDKey in $SPARKLE_KEYS does not match the keychain's Sparkle key" >&2
    exit 1
  }
fi
if [[ -f $DISK_IMAGE_SETTINGS ]]; then
  DMGBUILD_DIR=$HOME/Library/Caches/mac_release/dmgbuild-$DMGBUILD_VERSION
  if [[ ! -x $DMGBUILD_DIR/bin/dmgbuild ]]; then
    PYTHON=""
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$candidate" >/dev/null &&
        "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        PYTHON=$candidate
        break
      fi
    done
    [[ -n $PYTHON ]] || {
      echo "dmgbuild needs Python 3.10 or newer on PATH" >&2
      exit 1
    }
    rm -rf "$DMGBUILD_DIR"
    "$PYTHON" -m venv "$DMGBUILD_DIR"
    "$DMGBUILD_DIR/bin/pip" install --quiet --disable-pip-version-check --only-binary :all: \
      --require-hashes -r /dev/stdin <<<"$DMGBUILD_REQUIREMENTS"
  fi
fi
if [[ $UPLOAD == 1 && -n $RELEASE_BUCKET ]]; then
  # An R2 API token for the bucket, stored with
  # `security add-generic-password -s r2-<bucket> -a <access key ID> -w`.
  R2_ACCESS_KEY_ID=$(security find-generic-password -s "r2-$RELEASE_BUCKET" |
    sed -n 's/^ *"acct"<blob>="\(.*\)"$/\1/p')
  R2_SECRET_ACCESS_KEY=$(security find-generic-password -s "r2-$RELEASE_BUCKET" -w)
fi

# Builds the app with the given Bazel configs and stages it as $BUNDLE with the given profile.
# Sets VERSION and OUTPUT.
build_and_stage() {
  local channel=$1 profile=$2
  shift 2
  "$BAZEL" build "$@" --catalyst_cpus="$CPUS" "$TARGET"
  local unzipped=$HOME/Releases/.bazel-$ARCHIVE_NAME
  rm -rf "$unzipped"
  ditto -x -k "bazel-bin/$SOURCE/$BUILT_NAME.zip" "$unzipped"
  local built=$unzipped/$BUILT_NAME.app/Contents
  VERSION=$(plutil -extract CFBundleShortVersionString raw "$built/Info.plist")
  local bundle_version
  bundle_version=$(plutil -extract CFBundleVersion raw "$built/Info.plist")
  OUTPUT=$HOME/Releases/$ARCHIVE_NAME-$VERSION
  if [[ $channel == appstore ]]; then
    rm -rf "$OUTPUT"
  fi
  mkdir -p "$OUTPUT/$channel"
  local entry
  for entry in "$built"/*; do
    case ${entry##*/} in
      Info.plist | MacOS | PkgInfo | Resources | embedded.provisionprofile | _CodeSignature) ;;
      *)
        echo "Bazel's bundle has ${entry##*/}, which this script does not stage yet" >&2
        exit 1
        ;;
    esac
  done

  echo "Staging $APP_NAME $VERSION ($bundle_version) for $channel"
  BUNDLE=$OUTPUT/$channel/$APP_NAME.app
  mkdir -p "$BUNDLE/Contents/MacOS"
  ditto "$built/MacOS/$BUILT_NAME" "$BUNDLE/Contents/MacOS/$EXECUTABLE"
  ditto "$built/PkgInfo" "$BUNDLE/Contents/PkgInfo"
  ditto "$built/Resources" "$BUNDLE/Contents/Resources"
  local name
  for name in "${EXCLUDED_RESOURCES[@]+"${EXCLUDED_RESOURCES[@]}"}"; do
    rm -rf "$BUNDLE/Contents/Resources/$name"
  done
  ditto "$MAC_FILES/macOS.icns" "$BUNDLE/Contents/Resources/macOS.icns"
  cp "$MAC_FILES/Info.plist" "$BUNDLE/Contents/Info.plist"
  plutil -replace CFBundleShortVersionString -string "$VERSION" "$BUNDLE/Contents/Info.plist"
  plutil -replace CFBundleVersion -string "$bundle_version" "$BUNDLE/Contents/Info.plist"
  cp "$profile" "$BUNDLE/Contents/embedded.provisionprofile"
  rm -rf "$unzipped"
}

# Uploads through R2's S3 API, which takes single objects up to 5 GB; wrangler stops at 300 MiB.
r2_put() {
  local file=$1 key=$2 content_type=$3 cache_control=$4
  # R2 occasionally answers a large upload with 500; --retry resends on that and other
  # transient errors.
  curl -sS --fail-with-body --retry 3 -K - --aws-sigv4 "aws:amz:auto:s3" \
    -H "x-amz-content-sha256: UNSIGNED-PAYLOAD" \
    -H "Content-Type: $content_type" -H "Cache-Control: $cache_control" -T "$file" \
    "https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com/$RELEASE_BUCKET/$key" \
    <<<"user = \"$R2_ACCESS_KEY_ID:$R2_SECRET_ACCESS_KEY\""
}

sign() {
  local identity=$1
  shift
  codesign --force --timestamp --options runtime --sign "$identity" "$@"
}

# Submits a zip or disk image for notarization and waits; prints Apple's log when it is rejected.
notarize() {
  local result
  result=$(xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" --wait \
    --output-format json)
  if [[ $(plutil -extract status raw -o - - <<<"$result") != Accepted ]]; then
    echo "$result" >&2
    xcrun notarytool log "$(plutil -extract id raw -o - - <<<"$result")" \
      --keychain-profile "$NOTARY_PROFILE" >&2
    exit 1
  fi
}

build_and_stage appstore "$APP_STORE_PROFILE" --config=catalyst
sign "$APP_STORE_IDENTITY" --entitlements "$MAC_FILES/AppStore.entitlements" "$BUNDLE"
codesign --verify --strict --deep "$BUNDLE"
PACKAGE=$OUTPUT/$ARCHIVE_NAME.pkg
productbuild --product "$MAC_FILES/productDef.plist" --identifier "$BUNDLE_ID" --timestamp \
  --sign "$INSTALLER_IDENTITY" --component "$BUNDLE" /Applications "$PACKAGE"
rm -rf "$OUTPUT/appstore"

build_and_stage direct "$DEVELOPER_ID_PROFILE" --config=catalyst --config=direct
if [[ -f $SPARKLE_KEYS ]]; then
  /usr/libexec/PlistBuddy -c "Merge $SPARKLE_KEYS" "$BUNDLE/Contents/Info.plist"
  mkdir -p "$BUNDLE/Contents/Frameworks"
  ditto "$SPARKLE_DIR/Sparkle.framework" "$BUNDLE/Contents/Frameworks/Sparkle.framework"
  # Sparkle's documented order for re-signing its helpers: innermost first, never --deep.
  SPARKLE_CODE=$BUNDLE/Contents/Frameworks/Sparkle.framework/Versions/B
  sign "$DEVELOPER_ID_IDENTITY" "$SPARKLE_CODE/XPCServices/Installer.xpc"
  sign "$DEVELOPER_ID_IDENTITY" --preserve-metadata=entitlements \
    "$SPARKLE_CODE/XPCServices/Downloader.xpc"
  sign "$DEVELOPER_ID_IDENTITY" "$SPARKLE_CODE/Autoupdate"
  sign "$DEVELOPER_ID_IDENTITY" "$SPARKLE_CODE/Updater.app"
  sign "$DEVELOPER_ID_IDENTITY" "$BUNDLE/Contents/Frameworks/Sparkle.framework"
fi
sign "$DEVELOPER_ID_IDENTITY" --entitlements "$MAC_FILES/Direct.entitlements" "$BUNDLE"
codesign --verify --strict --deep "$BUNDLE"
# ditto keeps the framework's Versions/Current symlinks, which zip -r would flatten.
ditto -c -k --sequesterRsrc --keepParent "$BUNDLE" "$OUTPUT/notarize.zip"
notarize "$OUTPUT/notarize.zip"
rm "$OUTPUT/notarize.zip"
xcrun stapler staple "$BUNDLE"
spctl --assess --type execute "$BUNDLE"
ditto -c -k --sequesterRsrc --keepParent "$BUNDLE" "$OUTPUT/archive.zip"
ARCHIVE=$OUTPUT/$ARCHIVE_NAME-$VERSION-$(shasum -a 256 "$OUTPUT/archive.zip" | cut -c1-8).zip
mv "$OUTPUT/archive.zip" "$ARCHIVE"
DISK_IMAGE=""
if [[ -f $DISK_IMAGE_SETTINGS ]]; then
  # The disk image holds the stapled app, so a first launch from it needs no network check either.
  "$DMGBUILD_DIR/bin/dmgbuild" -s "$DISK_IMAGE_SETTINGS" -D app="$BUNDLE" "$APP_NAME" \
    "$OUTPUT/disk.dmg"
  codesign --force --timestamp --sign "$DEVELOPER_ID_IDENTITY" "$OUTPUT/disk.dmg"
  notarize "$OUTPUT/disk.dmg"
  xcrun stapler staple "$OUTPUT/disk.dmg"
  spctl --assess --type open --context context:primary-signature "$OUTPUT/disk.dmg"
  DISK_IMAGE=$OUTPUT/$ARCHIVE_NAME-$VERSION-$(shasum -a 256 "$OUTPUT/disk.dmg" | cut -c1-8).dmg
  mv "$OUTPUT/disk.dmg" "$DISK_IMAGE"
fi
rm -rf "$OUTPUT/direct"
if [[ -f $SPARKLE_KEYS ]]; then
  FEED_URL=$(plutil -extract SUFeedURL raw "$SPARKLE_KEYS")
  mkdir -p "$OUTPUT/appcast"
  ln "$ARCHIVE" "$OUTPUT/appcast/"
  # Only generate_keys, which created the keychain item, reads the key without an Allow prompt,
  # and SSH sessions cannot show that prompt. It exports the key to a private folder for
  # generate_appcast, which deletes it afterwards.
  KEY_DIR=$(mktemp -d)
  trap 'rm -rf "$KEY_DIR"' EXIT
  "$SPARKLE_DIR/bin/generate_keys" -x "$KEY_DIR/ed25519" >/dev/null
  "$SPARKLE_DIR/bin/generate_appcast" --ed-key-file "$KEY_DIR/ed25519" --maximum-deltas 0 \
    --download-url-prefix "${FEED_URL%/*}/" -o "$OUTPUT/appcast.xml" "$OUTPUT/appcast"
  rm -rf "$KEY_DIR" "$OUTPUT/appcast"
fi

if [[ $UPLOAD == 1 ]]; then
  if [[ -n $RELEASE_BUCKET ]]; then
    IMMUTABLE="public, max-age=31536000, immutable"
    # Files go up before appcast.xml so the feed never points at a missing archive.
    r2_put "$ARCHIVE" "${ARCHIVE##*/}" application/zip "$IMMUTABLE"
    if [[ -n $DISK_IMAGE ]]; then
      r2_put "$DISK_IMAGE" "${DISK_IMAGE##*/}" application/x-apple-diskimage "$IMMUTABLE"
      # A stable name for download links, replaced by each release.
      r2_put "$DISK_IMAGE" "$ARCHIVE_NAME.dmg" application/x-apple-diskimage "public, max-age=300"
    fi
    if [[ -f $SPARKLE_KEYS ]]; then
      r2_put "$OUTPUT/appcast.xml" appcast.xml application/xml "public, max-age=300"
    fi
  fi
  xcrun altool --upload-app -f "$PACKAGE" -t macos --apiKey "$ASC_API_KEY" \
    --apiIssuer "$ASC_API_ISSUER"
fi
echo "App Store package: $PACKAGE"
echo "Direct download: ${DISK_IMAGE:-$ARCHIVE}"
