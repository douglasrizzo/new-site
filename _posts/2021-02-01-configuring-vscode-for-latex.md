---
layout: post
title: Configuring Visual Studio Code for LaTeX
categories: vscode
---

VS Code has very poor LaTeX support out-of-the-box. Here I list useful extensions and other software that will transform VS Code into a much more power LaTeX editor. They also do not conflict with each other, which is great.

## Getting it out of the way: LaTeX Workshop

While I have used LaTeX Workshop for a long time, I started experiencing a few issues with it that, over time, got really annoying. The main issue is simply [the Enter key not working](https://github.com/James-Yu/LaTeX-Workshop/issues/1193), which was then replaced by [a substantial delay between pressing the Enter key and actually inserting a new line character](https://github.com/James-Yu/LaTeX-Workshop/issues/903). Fixing one apparently introduced the other, so, after many tries, I just gave up on LaTeX Workshop.

## What I use now

* Syntax highlighting [Rich LaTeX syntax highlighting (for use with Texlab)](https://marketplace.visualstudio.com/items?itemName=vomout.latex-syntax)
* Grammar checking: [LanguageTool for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=adamvoss.vscode-languagetool) + one of the multiple [language support extensions](https://marketplace.visualstudio.com/search?term=LanguageTool&target=VSCode) + [LTeX for LaTeX/Markdown support](https://marketplace.visualstudio.com/items?itemName=valentjn.vscode-ltex)
* Code completion and other shenanigans: [TeXLab](https://texlab.netlify.app/) + its [VS Code extension](https://marketplace.visualstudio.com/items?itemName=efoerster.texlab)
* PDF visualization: [Open](https://marketplace.visualstudio.com/items?itemName=sandcastle.vscode-open) to open a file in its default application. While there are extensions that explicitly give VS Code the capabilities of opening PDF files with PDF.js, I find PDF.js too simple and prefer just using Acrobat or Okular.
