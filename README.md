# Draw Things Community

This is a community repository that maintains public-facing code that runs [the Draw Things app](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820).

Currently, it contains the source code for our re-implementation of image generation models, samplers, data models, trainer within the app. Over time, as we move more core functionalities into separate libraries, this repository will grow.

# Contributions

While we expect the development to be mainly carried out by us, we welcome contributions from the community. We require a Contributor License Agreement to make this more manageable. If you prefer not to sign the CLA, you're still welcome to fork this repository and contribute in your own way.

# Roadmap and Repository Sync

The Draw Things app is managed through a private mono-repository. We do per-commit sync with the community repository in both ways. Thus, internal implemented features would be "leaked" to the community repository from time to time and it is expected.

That being said, we don't plan to publish roadmap for internal mono-repository. Whether we would like to publish roadmap for public community repository would require further deliberation and feedback from the community.

# License

This repository is licensed under GNU General Public License version 3 (GPL-v3) just like [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) or [Fooocus](https://github.com/lllyasviel/Fooocus/).

## License Examples

To illustrate how GPL-v3 applies in various scenarios, consider the following examples of acceptable licensing practices under our policy:

### Standalone Applications

If you develop a new application using code from this repository, and you release it on any app distribution platform (e.g., App Store) as a standalone product (meaning it functions independently and is not merely an HTTP client for Draw Things), you are required to license the entire application under GPL-v3.

### Forking and Enhancements

Should you fork this repository and introduce enhancements or additional functionalities, any new code you contribute must also be licensed under GPL-v3. This ensures all derivative works remain open and accessible under the same terms.

### Server and Client Development

If you use this repository to build a server application, either through HTTP or Google Protobuf (not limited to either), and subsequently develop a client application that communicates with your server, the following rules apply:

 * The source code for both the Google Protobuf definitions and the server must be published under GPL-v3.
 * The client application can be licensed under a different license, provided you do not bundle the client and server into a single distribution package or facilitate the automatic download of the server from within the client (a practice often referred to as "deep integration"). This distinction ensures that while the server-side components remain open, developers have flexibility in licensing client-side applications.

These examples are meant to provide guidance on common use cases involving our codebase. By adhering to these practices, you help maintain the spirit of open collaboration and software freedom that GPL-v3 champions.

## Alternative Licenses

We can provide alternative licenses for this repo, but not any 3rd-party forks. In particular, we may provide [LGPL-v3](https://www.gnu.org/licenses/lgpl-3.0.en.html) license as an alternative to free software on case-by-case basis. If you want to acquire more liberal licenses for closed-source applications or other legal obligations, please contact us. 
