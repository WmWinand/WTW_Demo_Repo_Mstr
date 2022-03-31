/*
auditor.sas
This code will audit a SAS data set.

The measurements we will be taking are:
number of observations
number of variables
modification date

This code is example code and comes with no warranties :)
*/

%let pathname = " c:\gitdemos\gitdata";
/*%let pathname = "/home/sasdemo/GitData";*/

%macro create_test_data; 
    libname test &pathname; 

    data test.test_data; 
      do i = 1 to 10; 
        output; 
      end; 
    run; 

%mend create_test_data; 

%create_test_data;

proc print data=test.test_data;
run;

/* Gather the measurements for our data set */
libname target &pathname; 

data measurements; 
  dsid = open('test.test_data'); 
  nobs = attrn(dsid,'nobs'); 
  nvars = attrn(dsid,'nvars'); 
  modte = datepart( attrn(dsid,'modte') ); * no need for the time; 
      rc = close(dsid); 
      audit_date = today(); 
     drop rc dsid; 
run;

proc print data=measurements;
  run;


