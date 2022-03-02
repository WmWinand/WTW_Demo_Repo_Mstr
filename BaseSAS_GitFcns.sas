/*******************************************/
/* Program with Base SAS Git functions for */
/* WTW_Demo_Repo                           */
/*******************************************/

data _null_;
    n = git_status("/home/sasdemo/WTW_Demo_Repo");
    put n=;
run;

data _null_;
   n = git_status("/home/sasdemo/WTW_Demo_Repo");                                      															/**/
   rc = git_index_add("/home/sasdemo/WTW_Demo_Repo", "BaseSAS_GitFcns.sas", "file1-status");   /**/
   rc = git_status_free("/home/sasdemo/WTW_Demo_Repo");
   n = git_status("/home/sasdemo/WTW_Demo_Repo");                                      															/**/
run;

data _null_;
 rc = git_commit(
    "/home/sasdemo/WTW_Demo_Repo",
    "HEAD",
    "William Winand",
    "t.winand@sas.com",
    "Updated BaseSAS_GitFcns with new functions");           /**/
   put rc=;
run;

/* Git Push Using SSH Keys */
data _null_;
 rc= git_push(
  "/home/sasdemo/WTW_Demo_Repo",
  "git",
  "",
  "/home/sasdemo/SSH_Keys/id_rsa.pub",
  "/home/sasdemo/SSH_Keys/id_rsa");
run;

/* Git Push Using HTTPS */