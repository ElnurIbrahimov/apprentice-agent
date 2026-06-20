describe('newtab recent conversation markup', () => {
  it('should not contain nested button elements', () => {
    const html = `<div class="nt-recent-item" data-conv-id="test">
      <span class="nt-recent-item-icon">icon</span>
      <span class="nt-recent-item-text">Title</span>
      <span class="nt-recent-item-time">5m ago</span>
      <button class="nt-recent-item-delete" title="Remove">x</button>
    </div>`;
    const buttonCount = (html.match(/<button/g) || []).length;
    expect(buttonCount).toBe(1);
    expect(html).not.toMatch(/<button[^>]*>[\s\S]*<button/);
  });
});