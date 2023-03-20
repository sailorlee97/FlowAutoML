#!/bin/sh

dos2unix features

> tmp-code
while read line;do

  # echo "grep $line features-code"
  grep $line features-code > tmp-code-1
  if [ $? -eq 0 ];then
	num=`cat tmp-code-1 | wc -l`
	#echo "num=$num"
	if [ $num -eq 1 ];then
		cat tmp-code-1 >> tmp-code
	else
		echo "$line found 2 or more lines!"
	fi
  else
	echo "$line not found!"
  fi

done < features

echo '#ifndef _NORMALIZED_FIELDS_H' > normalizedfields.h
echo '#define _NORMALIZED_FIELDS_H' >> normalizedfields.h
echo '#define GET_NORMALIZED_FIELDS \' >> normalizedfields.h
linenumber=0
awk -F[ '{print $1}' tmp-code > tmp-code-1
while read line;do
	#echo $linenumber $line
	echo "${line}[$linenumber]) \\" >> normalizedfields.h
	linenumber=`expr $linenumber + 1`
done < tmp-code-1

echo "" >> normalizedfields.h

echo "#define FEATURE_TOTAL_NUMBER $linenumber" >> normalizedfields.h
echo "" >> normalizedfields.h
echo '#endif' >> normalizedfields.h

echo "done $linenumber"

