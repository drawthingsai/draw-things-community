# Draw Things Community

This is a community repository that maintains public-facing code that runs [Draw Things the app](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820).

Currently, it contains the source code for our re-implementation of image generation models, samplers, data models, trainer within the app. Over time, as we move more core functionalities into separate libraries, this repository will grow.

# Contributions

While we expect the development to be mainly carried out by us, we welcome contributions from the community. We do leverage Contributor License Agreement to make this more manageable. If you don't like to sign the CLA, you are welcome to fork this repository.

# Roadmap and Repository Sync

Draw Things the app is managed through a private mono-repository. We do per-commit sync with the community repository in both ways. Thus, internal implemented features would be "leaked" to the community repository from time to time and it is expected.

That has been said, we don't plan to publish roadmap for internal mono-repository. Whether we would like to publish roadmap for public community repository would require further deliberation and feedback from the community.

# License

This repository is licensed under GPL-v3.

## License Examples

Given that we use GPL-v3 as the base license (just like [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [Fooocus](https://github.com/lllyasviel/Fooocus/) and [SD-Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)(AGPL-v3)), we would like to give some examples of what is, in our view, acceptable license practices:

 1. If you created a new app with code inside this repo, and published it in AppStore, and this new app can function independently (i.e. not a HTTP client to Draw Things), you need to publish the said app as a whole under GPL-v3 license.

 2. If you forked this repository, added enhancements for the functionalities of this repo, the newly added code need to be published under GPL-v3 license.

 3. If you created a HTTP / Google Protobuf server from this repo, and created a HTTP client that talks to this server, your Google Protobuf source code and server code needs to be published under GPL-v3. The HTTP client can be published under different licenses as long as you don't package both the client and the server in one deliverable or provide streamlined experience to download thee server from client automatically (as known as "deep integration").

## Alternative Licenses

We can provide alternative licenses for this repo, but not any 3rd-party forks. In particular, we may provide [LGPL-v3](https://www.gnu.org/licenses/lgpl-3.0.en.html) license as an alternative to free software on case-by-case basis. If you want to acquire more liberal licenses for close-source applications or other legal obligations, please contact us. 
