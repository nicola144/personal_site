# personal_site

Personal academic website built with Hugo and deployed to GitHub Pages.

## Local development

Install the Hugo extended release used by the deployment workflow (`0.128.0`), then run:

```sh
hugo server -D
```

For a production-style local build:

```sh
hugo --minify --destination /tmp/personal-site-build
```

The site source lives in `content/`, `layouts/`, `static/`, and `data/`. The `public/` directory is generated output and is not edited directly.

## Deployment

Pushes to `main` run `.github/workflows/hugo.yml`, which builds the site and deploys the generated artifact to GitHub Pages.
