---
layout: default
title: {{ site.name }}
---

---
# Table of contents
* [What is Deformable Generator?](#what_is)
* [Experimental results](#experimental_results)
* [Conclusion](#conclusion)
---

<a name="what_is"></a>

# What is Deformable Generator?
The deformable generator model is a deep generative model which disentangles the appearance and geometric information for both image and video data in purely unsupervised manner. The attributes of the visual data can be summarized as appearance (mainly including color, illumination, identity or category) and geometry (mainly including viewing angle and shape). 

The deformable generator model contains two generators, the appearance generator network models the appearance related information, while the geometric generator network produces the deformable fields (displacement of the coordinate of each pixel). The two generator networks are combined by the geometric related warping, such as rotation and stretching, to obtain the final image or video sequences. 

Two generators act upon independent latent factors to extract disentangled appearance and geometric information from image or video sequences (The nonlinear transition model is introduced to both the appearance and geometric generators to capture to dynamic information for the spatial-temporal process in the video sequences). 

![deformable generator model]({{ site.baseurl }}/fig/framwork1.png)
<center><em>An illustration of the proposed model</em></center>

The model can be expressed as
$$
  begin{eqnarray}
    X &=&F(Z^a,Z^g; \theta)\\
    &=& F_w(F_a(Z^a;\theta_a),F_g(Z^g;\theta_g)) + \epsilon
  \end{eqnarray}
$$
# Inference and learning

# Experimental results






## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/andyxingxl/Deformable-generator/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/andyxingxl/Deformable-generator/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
