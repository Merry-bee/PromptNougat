import re

def SyncTexJs(pdfsyncBody):
    unit = 65781.76
    numberPages = 0
    currentPage = {}
    currentElement = {}

    latexLines = {}
    blockNumberLine = {}
    hBlocks = []

    files = {}
    pages = {}
    pdfsyncObject = {
        'offset': {'x': 0, 'y': 0},
        'version': '',
        'files': {},
        'pages': {},
        'blockNumberLine': {},
        'hBlocks': [],
        'numberPages': 0
    }

    if pdfsyncBody is None:
        return pdfsyncObject

    lineArray = pdfsyncBody.split("\n")

    pdfsyncObject['version'] = lineArray[0].replace('SyncTeX Version:', '')

    inputPattern = re.compile(r'Input:([0-9]+):(.+)')
    offsetPattern = re.compile(r'(X|Y) Offset:([0-9]+)')
    openPagePattern = re.compile(r'\{([0-9]+)$')
    closePagePattern = re.compile(r'\}([0-9]+)$')
    verticalBlockPattern = re.compile(r'\[([0-9]+),([0-9]+):(-?[0-9]+),(-?[0-9]+):(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)')
    closeverticalBlockPattern = re.compile(r'\]$')
    horizontalBlockPattern = re.compile(r'\(([0-9]+),([0-9]+):(-?[0-9]+),(-?[0-9]+):(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)')
    closehorizontalBlockPattern = re.compile(r'\)$')
    elementBlockPattern = re.compile(r'(.)([0-9]+),([0-9]+):-?([0-9]+),-?([0-9]+)(:?-?([0-9]+))?')

    for i in range(1, len(lineArray)):
        line = lineArray[i]

        # input files
        match = inputPattern.match(line)
        if match:
            files[match.group(1)] = {
                'path': match.group(2),
                'name': match.group(2).replace(r'^.*[\\\/]', '')
            }
            continue

        # offset
        match = offsetPattern.match(line)
        if match:
            pdfsyncObject['offset'][match.group(1).lower()] = int(match.group(2)) / unit
            continue

        # new page
        match = openPagePattern.match(line)
        if match:
            currentPage = {
                'page': int(match.group(1)),
                'blocks': [],
                'type': 'page'
            }
            if currentPage['page'] > numberPages:
                numberPages = currentPage['page']
            currentElement = currentPage
            continue

        # close page
        match = closePagePattern.match(line)
        if match:
            pages[match.group(1)] = currentPage
            currentPage = None
            continue

        # new V block
        match = verticalBlockPattern.match(line)
        if match:
            s1 = [int(match.group(3)) / unit, int(match.group(4)) / unit]
            s2 = [int(match.group(5)) / unit, int(match.group(6)) / unit]
            block = {
                'type': 'vertical',
                'parent': currentElement,
                'fileNumber': int(match.group(1)),
                'file': files[match.group(1)],
                'line': int(match.group(2)),
                'left': s1[0],
                'bottom': s1[1],
                'width': s2[0],
                'height': s2[1],
                'depth': int(match.group(7)),
                'blocks': [],
                'elements': [],
                'page': currentPage['page']
            }
            currentElement = block
            continue

        # close V block
        match = closeverticalBlockPattern.match(line)
        if match:
            if currentElement['parent'] is not None:
                currentElement['parent']['blocks'].append(currentElement)
                currentElement = currentElement['parent']
            continue

        # new H block
        match = horizontalBlockPattern.match(line)
        if match:
            s1 = [int(match.group(3)) / unit, int(match.group(4)) / unit]
            s2 = [int(match.group(5)) / unit, int(match.group(6)) / unit]
            block = {
                'type': 'horizontal',
                'parent': currentElement,
                'fileNumber': int(match.group(1)),
                'file': files[match.group(1)],
                'line': int(match.group(2)),
                'left': s1[0],
                'bottom': s1[1],
                'width': s2[0],
                'height': s2[1],
                'blocks': [],
                'elements': [],
                'page': currentPage['page']
            }
            hBlocks.append(block)
            currentElement = block
            continue

        # close H block
        match = closehorizontalBlockPattern.match(line)
        if match:
            if currentElement['parent'] is not None:
                currentElement['parent']['blocks'].append(currentElement)
                currentElement = currentElement['parent']
            continue

        # new element
        match = elementBlockPattern.match(line)
        if match:
            
            type_ = match.group(1)
            fileNumber = int(match.group(2))
            lineNumber = int(match.group(3))
            left = int(match.group(4)) / unit
            bottom = int(match.group(5)) / unit
            width = int(match.group(7)) / unit if match.group(7) else None


            elem = {
                'type': type_,
                'parent': currentElement,
                'fileNumber': fileNumber,
                'file': files[str(fileNumber)],
                'line': lineNumber,
                'left': left,
                'bottom': bottom,
                'height': currentElement['height'],
                'width': width,
                'page': currentPage['page']
            }
            if elem['file']['name'] not in blockNumberLine:
                blockNumberLine[elem['file']['name']] = {}
            if lineNumber not in blockNumberLine[elem['file']['name']]:
                blockNumberLine[elem['file']['name']][lineNumber] = {}
            if elem['page'] not in blockNumberLine[elem['file']['name']][lineNumber]:
                blockNumberLine[elem['file']['name']][lineNumber][elem['page']] = []
            blockNumberLine[elem['file']['name']][lineNumber][elem['page']].append(elem)
            if currentElement['elements'] is not None:
                currentElement['elements'].append(elem)
            continue

    pdfsyncObject['files'] = files
    pdfsyncObject['pages'] = pages
    pdfsyncObject['blockNumberLine'] = blockNumberLine
    pdfsyncObject['hBlocks'] = hBlocks
    pdfsyncObject['numberPages'] = numberPages
    return pdfsyncObject


if __name__ == '__main__':
    with open('data/tmp/0018origin.synctex','r') as fi:
        pdfsyncBody = fi.read()
    res=SyncTexJs(pdfsyncBody)   
    linebox = {}
    lines = res['blockNumberLine']['C:\\Users\\Zhong Han-Sen\\Documents\\latex\\0018origin.tex'].keys()
    for line in lines:
        pages = res['blockNumberLine']['C:\\Users\\Zhong Han-Sen\\Documents\\latex\\0018origin.tex'][line].keys()
        for page in pages:
            for block in res['blockNumberLine']['C:\\Users\\Zhong Han-Sen\\Documents\\latex\\0018origin.tex'][line][page]:
                left = block['left']
                bottom = block['bottom']
                width = block['width']
                height = block['height']
                if width is None or height is None:
                    continue
                if width < 0.1 or height < 0.1:
                    continue
                bbox = [left, bottom, left+width, bottom+ height]
                if (line, page) not in linebox:
                    linebox[(line, page)] = []
                linebox[(line, page)].append(bbox)   
    print(linebox)
                       