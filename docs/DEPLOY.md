# Deploying solve-vision-with-code.com

The site is this `docs/` folder, served twice from the same commit of branch `public`:

| Where | How it updates |
|---|---|
| https://solve-vision-with-code.com/ (nginx on a small box, root = a clone of this repo) | `sudo svc-update` on the box (`git pull --ff-only` + chown) |
| https://hokindeng.github.io/solve-vision-with-code/ (GitHub Pages, `public:/docs`) | automatically on push |

Numbers and data files (`data/`, `media/`, `paper.pdf`) never change in a deploy; only the presentation does.
Everything below is run from the repository root by the person deploying, one step at a time.

## 0. Gate

```sh
python3 docs/verify-site.py            # must end with "0 fail"
```

It checks banned strings, every link, `data/tasks.json` / `data/leaderboard.json` consistency, the canonical URLs and
the cite block, and that the last five `docs/` commits carry no `Co-Authored-By` trailer. Fix, do not skip.

## 1. Commit on `public`

Only `docs/` and one line of `README.md` change. No trailer of any kind.

```sh
git -C . status --short                # expect: docs/… and README.md only
git -C . add docs README.md
git -C . commit -m "site: redesign at Hokin's request — task grid with inline per-instance panel, separate leaderboard page; data unchanged"
git -C . log -1 --format=%B | grep -ci co-authored-by   # must print 0
```

## 2. Push and wait for GitHub Pages

```sh
git -C . push origin public
SHA=$(git -C . rev-parse HEAD)
for i in $(seq 1 30); do
  gh api repos/hokindeng/solve-vision-with-code/pages/builds/latest --jq '.status + " " + .commit' | tee /dev/stderr | grep -q "built $SHA" && break
  sleep 10
done
curl -sIo /dev/null -w '%{http_code}\n' https://hokindeng.github.io/solve-vision-with-code/leaderboard.html   # 200
```

## 3. Pull on the box

The box has no fixed IP: its address is whatever the domain resolves to (the A record is kept in step with the
machine). Read it from DNS, never from memory:

```sh
SVC_IP=$(dig +short solve-vision-with-code.com | head -1); echo "$SVC_IP"
ssh -i ~/hokin/.aws/hokin-harness.pem ec2-user@"$SVC_IP" 'sudo svc-update'
```

If ssh times out, the machine may have restarted with a new address and DNS not yet moved. Look it up read-only,
by its tag, in the research account (the profile is named in the machine record in the harness `.aws/` folder):

```sh
aws --profile <research-profile> ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Task,Values=solve-vision-with-code.com" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text
```

A changed address means the GoDaddy A record must follow — that is Hokin's step, not the deployer's.

## 4. Verify live

```sh
python3 docs/verify-site.py --live https://solve-vision-with-code.com
```

This confirms TLS (https 200, http 301), that the domain and GitHub Pages serve identical `index.html`,
`leaderboard.html`, `app.js`, `leaderboard.js`, `style.css`, that all five equal the local files, and that 200
random media and program files answer 200.

## Rollback

```sh
git -C . revert --no-edit HEAD          # keeps history; no trailer
git -C . push origin public
ssh -i ~/hokin/.aws/hokin-harness.pem ec2-user@"$(dig +short solve-vision-with-code.com | head -1)" 'sudo svc-update'
python3 docs/verify-site.py --live https://solve-vision-with-code.com
```

GitHub Pages rebuilds from the revert on its own (step 2's wait loop applies).

## Never

Stop or restart the box, touch DNS or the certificate, force-push `public`, or delete anything — old files are moved
under an `archive/` folder, not removed.
