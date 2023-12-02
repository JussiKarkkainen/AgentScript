git ls-files | xargs wc -l | awk '
    !/total$/ {
        print;
        split($2, path, "/");
        dir = (length(path) > 1) ? path[1] : ".";
        lines[dir] += $1;
        totalLines += $1;
    }
    END {
        print "\nLines per directory:";
        for (d in lines) {
            print lines[d], d;
        }
        print "\nTotal lines of code:", totalLines;
    }
'

