let SharedScript = """
    console.log = (...args) => {
      __dtHooks.log(args[0]);
    }

    console.warn = (...args) => {
      __dtHooks.log(args[0], 1);
    }

    console.error = (...args) => {
      __dtHooks.log(args[0], 2);
    }

    const MaskValueType = {
      PURE_NOISE: 1,
      MASK: 2,
      RETAIN_OR_MASK: 0
    }

    const SamplerType = {
      DPMPP_2M_KARRAS: 0,
      EULER_A: 1,
      DDIM: 2,
      PLMS: 3,
      DPMPP_SDE_KARRAS: 4,
      UNI_PC: 5,
      LCM: 6,
      EULER_A_SUBSTEP: 7,
      DPMPP_SDE_SUBSTEP: 8,
      TCD: 9,
      EULER_A_TRAILING: 10,
      DPMPP_SDE_TRAILING: 11
    }

    class Point {
      constructor(x, y) {
        this.x = x;
        this.y = y;
      }
    }

    class Size {
      constructor(width, height) {
        this.width = width;
        this.height = height;
      }
    }

    class Rectangle {
      constructor(x, y, width, height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
      }

      maxX() {
        return this.x + this.width;
      }
      maxY() {
        return this.y + this.height;
      }

      static union(rect1, rect2) {
        const x1 = Math.min(rect1.x, rect2.x);
        const y1 = Math.min(rect1.y, rect2.y);
        const x2 = Math.max(rect1.maxX(), rect2.maxX());
        const y2 = Math.max(rect1.maxY(), rect2.maxY());
        return new Rectangle(x1, y1, x2 - x1, y2 - y1);
      }

      contains(rect) {
        return this.maxX() > rect.maxX() && this.maxY() > rect.maxY() && this.x < rect.x && this.y < rect.y;
      }

      intersect(rect) {
        const x1 = Math.max(this.x, rect.x);
        const y1 = Math.max(this.y, rect.y);
        const x2 = Math.min(this.x + this.width, rect.x + rect.width);
        const y2 = Math.min(this.y + this.height, rect.y + rect.height);

        if (x2 > x1 && y2 > y1) {
          const intersectionX = x1;
          const intersectionY = y1;
          const intersectionWidth = x2 - x1;
          const intersectionHeight = y2 - y1;

          return new Rectangle(intersectionX, intersectionY, intersectionWidth, intersectionHeight);
        } else {
          // No intersection
          return null;
        }
      }

      exclude(rect) {
        // no exclusion if this contains the other rectangle
        if (this.contains(rect)) return this;
        const intersectionRect = this.intersect(rect);
        if (!intersectionRect) return this;
        const remainingRect = {
          x: this.x,
          y: this.y,
          width: this.width,
          height: this.height
        };

        if (intersectionRect.x === this.x && intersectionRect.maxX() != this.maxX()) {
          remainingRect.x += intersectionRect.width;
          remainingRect.width -= intersectionRect.width;
        } else if (intersectionRect.x != this.x && intersectionRect.maxX() === this.maxX()) {
          remainingRect.width -= intersectionRect.width;
        }

        if (intersectionRect.y === this.y && intersectionRect.maxY() != this.maxY()) {
          remainingRect.y += intersectionRect.height;
          remainingRect.height -= intersectionRect.height;
        } else if (intersectionRect.y != this.y && intersectionRect.maxY() === this.maxY()) {
          remainingRect.height -= intersectionRect.height;
        }
        return new Rectangle(remainingRect.x, remainingRect.y, remainingRect.width, remainingRect.height);
      }

      scale(factor) {
        this.width = Math.floor(this.width * factor);
        this.height = Math.floor(this.height * factor);
        this.x = Math.floor(this.x * factor);
        this.y = Math.floor(this.y * factor);
      }
    }

    class RNG {
      constructor(seed) {
        this.state = seed;
      }

      // Perform xorshift like in Swift. The results won't necessarily be the same though, because of 32-bit signed handling
      next() {
        var x = this.state == 0 ? 0xbad5eed : this.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        this.state = Math.abs(x);  // JS uses 32-bit signed ints, so convert it to unsigned for the app
        return this.state;
      }

      nextNumber() { // Backward compatibility.
        return this.next();
      }
    }

    // The ObjC-bridged object with the functions is finnicky. Its functions can't be found with Object.keys,
    // and it's tricky to override functions on the object, or to call those functions as free functions.
    // Since it's good to decouple the behavior of the functions that consumers call from the limitations
    // of the bridging API, e.g. the inability for Swift exceptions to propagate into JS, we have the
    // consumer call a separate object here

    const pipeline = {
      get configuration() {
        if (pipeline.__configuration) {
          return pipeline.__configuration;
        }
        const configuration = __dtHooks.existingConfiguration();
        pipeline.__configuration = configuration;
        return configuration;
      },
      get prompts() {
        return __dtHooks.existingPrompts();
      },
      run(args) {
        if (args.configuration) {
          pipeline.__configuration = args.configuration;
        }
        return __dtHooks.generateImage(args);
      },
      downloadBuiltin(filename) {
        return __dtHooks.downloadBuiltin(filename)
      },
      findControlByName(name) {
        return __dtHooks.createControl(name);
      },
      findLoRAByName(name) {
        return __dtHooks.createLoRA(name);
      }
    };

    const filesystem = {
      pictures: {
        readEntries(directory) {
          if (directory) {
            return __dtHooks.listFilesUnderPicturesWithinDirectory(directory)
          } else {
            return __dtHooks.listFilesUnderPictures()
          }
        },
        get path() {
          return __dtHooks.picturesPath()
        }
      }
    };

    const canvas = {
      get currentMask() {
        return __dtHooks.currentMask();
      },
      get foregroundMask() {
        const handle = __dtHooks.createForegroundMask();
        if (handle == 0) {
          return null;
        }
        return new Mask(handle);
      },
      get backgroundMask() {
        const handle = __dtHooks.createBackgroundMask();
        if (handle == 0) {
          return null;
        }
        return new Mask(handle);
      },
      get topLeftCorner() {
        return __dtHooks.topLeftCorner();
      },
      get boundingBox() {
        return __dtHooks.boundingBox();
      },
      get canvasZoom() {
        return __dtHooks.canvasZoom();
      },
      set canvasZoom(zoomScale) {
        __dtHooks.setCanvasZoom(zoomScale);
      },
      moveCanvas(left, top) {
        return __dtHooks.moveCanvas(left, top);
      },
      moveCanvasToRect(rect) {
        const origin = rect.origin;
        const size = rect.size;
        canvas.moveCanvas(origin.x, origin.y);
        const configuration = pipeline.configuration;
        // rect is always a square for face detection
        const zoom = Math.min(configuration.width, configuration.height) / size.width;
        canvas.canvasZoom = zoom;
      },
      updateCanvasSize(configuration) {
        return __dtHooks.updateCanvasSize(configuration);
      },
      createMask(width, height, value) {
        const handle = __dtHooks.createMask(width, height, value);
        return new Mask(handle);
      },
      clear() {
        __dtHooks.clearCanvas();
      },
      loadImage(file) {
        __dtHooks.loadImageFileToCanvas(file);
      },
      saveImage(file, visibleRegionOnly = false) {
        __dtHooks.saveImageFileFromCanvas(file, visibleRegionOnly);
      },
      saveImageSrc(visibleRegionOnly = false) {
        return __dtHooks.saveImageSrcFromCanvas(visibleRegionOnly);
      },
      loadMaskFromPhotos() {
        __dtHooks.loadLayerFromPhotos("mask")
      },
      loadDepthMapFromPhotos() {
        __dtHooks.loadLayerFromPhotos("depthMap")
      },
      loadScribbleFromPhotos() {
        __dtHooks.loadLayerFromPhotos("scribble")
      },
      loadPoseFromPhotos() {
        __dtHooks.loadLayerFromPhotos("pose")
      },
      loadColorFromPhotos() {
        __dtHooks.loadLayerFromPhotos("color")
      },
      loadCustomLayerFromPhotos() {
        __dtHooks.loadLayerFromPhotos("custom")
      },
      loadMoodboardFromPhotos() {
        __dtHooks.loadLayerFromPhotos("shuffle")
      },
      loadMaskFromFiles() {
        __dtHooks.loadLayerFromFiles("mask")
      },
      loadDepthMapFromFiles() {
        __dtHooks.loadLayerFromFiles("depthMap")
      },
      loadScribbleFromFiles() {
        __dtHooks.loadLayerFromFiles("scribble")
      },
      loadPoseFromFiles() {
        __dtHooks.loadLayerFromFiles("pose")
      },
      loadColorFromFiles() {
        __dtHooks.loadLayerFromFiles("color")
      },
      loadCustomLayerFromFiles() {
        __dtHooks.loadLayerFromFiles("custom")
      },
      loadMoodboardFromFiles() {
        __dtHooks.loadLayerFromFiles("shuffle")
      },
      loadMaskFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "mask")
      },
      loadDepthMapFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "depthMap")
      },
      loadScribbleFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "scribble")
      },
      loadPoseFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "pose")
      },
      loadColorFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "color")
      },
      loadCustomLayerFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "custom")
      },
      loadMoodboardFromSrc(srcContent) {
        __dtHooks.loadLayerFromSrc(srcContent, "shuffle")
      },
      loadPoseFromJson(jsonString) {
        __dtHooks.loadLayerFromJson(jsonString, "pose")
      },
      detectFaces() {
        return __dtHooks.detectFaces();
      },
    //  detectHands() {
    //    return __dtHooks.detectHands();
    //  },
    };

    class Mask {
      constructor(handle) {
        this.handle = handle;
      }

      fillRectangle(x, y, width, height, value) {
        __dtHooks.fillMaskRectangle(this, new Rectangle(x, y, width, height), value);
      }
    }

    function __dtPrettyPrint(obj) {
      return JSON.stringify(obj, null, 2);
    }

    function __dtSleep(seconds) {
      const milliseconds = seconds * 1000;
      const start = Date.now();
      while (Date.now() - start < milliseconds) {
        ;
      }
    }


    function requestFromUser(title, confirm, construction) {
      const widget = {
        size(width, height, minValue, maxValue) {
          return {"type": "size", "width": width, "height": height, "minValue": minValue, "maxValue": maxValue};
        },
        slider(value, valueType, minValue, maxValue, title) {
          return {"type": "slider", "title": title, "valueType": valueType, "value": value, "minValue": minValue, "maxValue": maxValue};
        },
        textField(value, placeholder) {
          return {"type": "textField", "value": value, "placeholder": placeholder};
        },
        imageField() {
          return {"type": "imageField"};
        },
        switch(isOn, title) {
          return {"type": "switch", "isOn": isOn, "title": title};
        },
        segmented(index, options) {
          return {"type": "segmented", "index": index, "options": options};
        },
        menu(index, options) {
          return {"type": "menu", "index": index, "options": options};
        },
        section(title, detail, views) {
          return {"type": "section", "title": title, "detail": detail, "views": views};
        },
        plainText(value) {
          return {"type": "plainText", "value": value};
        }
      }
      widget.slider.percent = {"percent": true}
      widget.slider.fractional = function(k) { return {"fractional": k} }
      widget.slider.scale = {"scale": true}
      return __dtHooks.requestFromUser(title, confirm, construction.call(widget));
    }

  """