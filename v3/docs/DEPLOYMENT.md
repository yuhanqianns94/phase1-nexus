# Deploying the public website

The website runs one Streamlit server with isolated visitor sessions. Visitors can
choose the shared library or upload their own literature and provide their own API
key and extraction model. The owner's API key is never configured on the public
server. Local development keeps its existing credentials and model defaults.

Entering an email address identifies the current browser session only. The address
is not verified, does not grant access to another session, and does not trigger an
email. Closing or losing the session does not provide a way to recover its private
uploads or analyses through that email address. Visitors should download results
before leaving. Session files are temporary and are cleaned up after the configured
retention period (24 hours by default).

## Server size and concurrency

The supplied `render.yaml` starts one paid service with 1 CPU, 2 GB RAM, and a
10 GB persistent disk for an initial deployment with one or two connected users.
Expensive scientific calculations share one worker; additional calculations wait
in the queue while visitors can continue browsing. Public BOIN jobs
are limited to 20,000 simulated trials per design and 50,000 trials across one
design comparison; local computation limits are unchanged.

A local benchmark on October 18, 2026 measured five retained public browser-test
sessions using three completed local articles (nine model scopes). The session
process peaked at 265.3 MiB. Separate uncached synthetic fits with one, two and
three drugs used the unchanged default sampler (750 warmup steps, 750 samples,
two chains); the largest worker peaked at 583.5 MiB. A 100-scenario by 100-trial
BOIN simulation also completed. Adding the separately measured session and
largest-worker peaks gives **848.8 MiB**, before deployment headroom.

**The initial configuration is 2 GB RAM / 1 CPU (`plan: 1c-2g`).** This is the
smallest trial size supported by the small benchmark. Monitor memory and job
latency after deploying the completed shared library. If more headroom is needed,
change the Blueprint to `plan: 2c-4g` for 4 GB RAM / 2 CPUs. The measurements above
come from macOS arm64 / Python 3.13; the deployment uses Linux / Python 3.12.
Re-run `v3/scripts/benchmark_hosting.py --output /tmp/hosting.json` with the
deployment interpreter or use Render's metrics to check actual resource use.

The initial disk allocation covers the owner's shared library, its published
caches, and small session results. It does not budget for a growing collection of
visitor uploads. Upload limits and temporary retention prevent that optional
feature from becoming a permanent literature archive. Measure the completed
owner library and allow room for both the old and new release during an update;
increase the disk if needed. Code and installed dependencies live in the container
image, separately from the persistent disk.

As checked on October 18, 2026, Render lists **$25/month for 1 CPU / 2 GB RAM**
and **$85/month for 2 CPUs / 4 GB RAM**. Persistent disks cost **$0.25/GB/month**,
so the 10 GB disk adds $2.50: approximately **$27.50/month for the configured size**
or **$87.50/month for the optional 4 GB upgrade**, before taxes and usage overages. A free
Hobby workspace can contain this paid compute service; a separate paid team
workspace is not required. See the [official pricing table](https://render.com/pricing)
and [workspace plans](https://render.com/docs/platform-features-by-plan).

Confirm the displayed charges in the [Render dashboard](https://dashboard.render.com/)
before approving creation. No paid resource is created by preparing or pushing
this repository.

## Register and deploy on Render

1. Open [Render](https://dashboard.render.com/register), create an account, and
   connect the GitHub account that owns `phase1-toxicity-v3`.
2. Add a payment method when Render requests one for the paid compute service.
3. Choose **New > Blueprint** and select the `phase1-toxicity-v3` repository and
   its `main` branch. Use the root-level `render.yaml`; leave the repository root
   directory empty. The service uses the root-level Dockerfile.
4. Review the resources: one web service, 1 CPU / 2 GB RAM, one 10 GB persistent
   disk mounted at `/var/data`, and the region shown in `render.yaml`. Approve
   creation only after reviewing the displayed charges.
5. Keep `PHASE1_DEPLOYMENT=public` and
   `PHASE1_DATA_ROOT=/var/data/phase1`. Do **not** add `OPENAI_API_KEY`,
   `OPENAI_BASE_URL`, or any local provider credential to this service.
6. Leave `PHASE1_SHARED_BUNDLE_URL` and `PHASE1_SHARED_BUNDLE_SHA256` empty for the
   first deployment. An unfinished local library is not published automatically.
   Add both variables later when a reviewed library release is ready.
7. Wait for the GitHub **Website checks** workflow and the Render build to finish.
   Open the `https://...onrender.com` URL shown on the service's Overview page.
   Render provides HTTPS. A custom domain can be added later.
8. Check the public website using two separate browsers: both can select the
   shared library, while their uploaded literature, drafts, results and API keys
   stay in separate sessions. A new installation may have an empty shared
   library until the first library bundle is installed.

The repository sets `autoDeployTrigger: checksPass`. After a later push to `main`,
Render waits for the GitHub checks and then deploys the new commit. A single
service with a persistent disk has a short interruption during deployment, and
active visitor sessions can be lost. Download important results before updating.
See [Render automatic deployments](https://render.com/docs/deploys) and
[persistent disk limitations](https://render.com/docs/disks).

## Publish local code changes

Keep editing and testing the original local v3 project as usual. Its shell
credentials and extraction model settings are unchanged. The independent public
checkout contains the source used by the website.

From the original project's root directory, preview an export:

```bash
v3/.venv/bin/python v3/scripts/publish_website.py --check
```

The command reports changed filenames and runs a secret scan. It does not modify
files, make a commit, or contact GitHub. It exports only application source, tests,
selected example files and this deployment guide. Both public READMEs remain
feature-only. Local literature, outputs, prediction caches, private documents,
virtual environments and credentials are excluded. The `.env.example` files are
replaced with empty templates.

To prepare files for review without publishing:

```bash
v3/.venv/bin/python v3/scripts/publish_website.py --stage
```

Review the diff in the independent public checkout. When the changes are ready,
run one command to scan, commit and push them:

```bash
v3/.venv/bin/python v3/scripts/publish_website.py --push \
  --message "Update the public website"
```

This uses the existing Git/GitHub authentication. It does not embed credentials
in a remote URL or a configuration file. Gitleaks is required for `--push`; the
script finds an installed `gitleaks` executable or the scanner from the original
publication. Alternatively provide `--gitleaks /absolute/path/to/gitleaks`.

The push updates the website automatically after CI passes. Merely saving a local
file does not publish an unfinished edit. The exporter never force-pushes or
silently merges a newer remote branch. If code was edited through GitHub or
Codespaces, reconcile those changes with the original local v3 source before
publishing again. `--checkout /absolute/path/to/public-checkout` selects a
replacement public checkout if the original one was moved.

## Publish the processed shared library later

Code deployment and library publication are separate operations. Finish extracting
and reviewing the owner's literature locally, then run the desired local fits to
populate the source-posterior cache. A source posterior can be reused only when
its evidence, model, prior, numerical runtime and fit settings match. Similarity
weights or prediction results that depend on a visitor's target are not universal
parameters for every visitor.

The bundle tool exports only the supported processed evidence and reusable cache
artifacts. It does not export API keys, request logs, raw extraction responses or
the whole local output directory. Raw article PDFs are not required for the shared
prediction library. Use only material you are entitled to redistribute publicly.

Create a release bundle, replacing paths with the completed extraction and
posterior directories:

```bash
v3/.venv/bin/python v3/scripts/publish_shared_library.py pack \
  --extraction-root /absolute/path/to/completed/extractions \
  --posterior-root /absolute/path/to/source_posteriors \
  --output /absolute/path/to/shared-library.tar.gz
```

Repeat `--extraction-root` or `--posterior-root` for additional directories. Review
the tool's report and keep the bundle's SHA256. Do not publish the current
unfinished library simply to populate the first deployment.

Upload the reviewed bundle as an asset on a GitHub release in the public repository
(**Releases > Draft a new release > Attach binaries**). The asset is public; this
step intentionally shares the included processed evidence and caches. Copy its
HTTPS download URL, normally in this form:

```text
https://github.com/OWNER/REPOSITORY/releases/download/TAG/shared-library.tar.gz
```

Set the following two variables in the Render service's Environment settings, then
save and deploy:

```text
PHASE1_SHARED_BUNDLE_URL=<the release asset HTTPS URL>
PHASE1_SHARED_BUNDLE_SHA256=<the bundle SHA256>
```

On startup the server downloads the bundle, verifies the digest, and installs it
under the persistent data directory. Releases have this layout:

```text
/var/data/phase1/shared/
  current.json
  releases/<release-id>/
    library/
    source_posteriors/
```

The current-release pointer is updated after installation succeeds. Keep the last
known-good bundle URL and checksum so you can select it again if needed. Publish
a new bundle after completing more literature or useful local cache runs; source
code updates alone do not copy private local data to the website. Do not manually
edit installed release contents.

The scientific dependency versions in `requirements-deploy.txt` match the local
cache-producing environment. Update those pins deliberately, run CI, and rebuild
or revalidate shared caches when changing the numerical stack. Cache files do not
make different CPU platforms mathematically identical; the application validates
its supported cache signatures before reusing results.

## Edit from another computer

Open the public GitHub repository and choose **Code > Codespaces > Create
codespace on main**. Its dev-container configuration installs Python 3.12 and the
pinned dependencies. Preview the ordinary development application with:

```bash
cd v3
python -m streamlit run phase1_web/app.py --server.address 0.0.0.0
```

To preview the public mode without owner credentials, start the hosted launcher
with an empty local data directory instead:

```bash
PHASE1_DEPLOYMENT=public PHASE1_DATA_ROOT=/tmp/phase1-public-preview \
  python -m phase1_web.server
```

Keep the forwarded development port private. Run `python -m pytest` from `v3`,
commit the reviewed changes, and push to `main`; the same CI and Render deployment
process applies. Codespaces has separate usage limits and billing. See
[GitHub dev containers](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers).
