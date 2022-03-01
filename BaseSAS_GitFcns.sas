/*******************************************/
/* Program with Base SAS Git functions for */
/* WTW_Demo_Repo                           */
/*******************************************/

data _null_;
    n = git_status("/home/sasdemo/WTW_Demo_Repo_SSH");
    put n=;
run;

data _null_;
 rc = git_commit(
    "/home/sasdemo/WTW_Demo_Repo_SSH",
    "HEAD",
    "William Winand",
    "t.winand@sas.com",
    "Updated BaseSAS_GitFcns with new functions");           /**/
   put rc=;
run;

data _null_;
 rc= git_push(
  "/home/sasdemo/WTW_Demo_Repo_SSH",
  "git",
  "",
  "/home/sasdemo/SSH_Keys/id_rsa.pub",
  "/home/sasdemo/SSH_Keys/id_rsa");
run;