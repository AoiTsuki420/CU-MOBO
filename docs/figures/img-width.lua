-- Scale every image to the page text width when converting to docx,
-- so wide matplotlib panels don't overflow the Word margins.
function Image(el)
  el.attributes.width = "100%"
  return el
end
