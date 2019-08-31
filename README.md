# mr_nlp

## Workflow
### First do this on the terminal
`git pull origin master` to pull changes from oracle.\
`git branch feature_branch` to make a new branch to work on.\
`git checkout feature_branch` to move to it.\

Go and make some changes...\
Do the normal local git workflow on your feature branch:\
`git add <some_files>`\
`git commit -m <a nice descriptive message>`\
Use `git push origin master` to publish your feature branch to github.\
If you had merged with your local master here to would get an error when you tried to push,
since commiting to the origin master is restricted.

### Now do this on github (can be done in terminal but more fiddly)
Go to "pull requests", and create a new pull request.\
Wait for the other person to review changes.\
Merge the changes, I prefer to use rebase.\
Use the github button to remove the feature branch from the origin repo.\
Alternatively you can do this in terminal with `git push origin --delete <feature_branch>`

### Back in the terminal, update your own repo
`git checkout master`
`git branch -d feauture_branch`
`git pull origin master`

And you're done.
