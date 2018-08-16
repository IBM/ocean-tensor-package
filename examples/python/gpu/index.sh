#!/bin/bash

for filename in `find . -name "*.py"`; do
   v=`grep "##" ${filename} | sed "s/^##//"`
   if [[ -z "$v" ]]; then
      printf "\033[1;32m%-38s\033[0m ---\n" ${filename}
   else
      i=0
      echo "${v}" | while read -r line; do
         i=$[i + 1]
         if [[ ( "$i" == 1 ) ]]; then
            printf "\033[1;32m%-38s\033[0m" ${filename}
         else
            printf "%-34s" " "
         fi
         printf " %s\n" "${line}"
      done
   fi
done

