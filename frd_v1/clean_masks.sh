find /path/to/your/dir -type f -iname "*mask*" -exec sh -c 'file --mime-type "$1" | grep -qE "image/" && rm "$1"' _ {} \;

